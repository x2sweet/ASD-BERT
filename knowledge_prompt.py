import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import jieba
from fuzzywuzzy import fuzz
import pandas as pd
import requests
import time
import os
import traceback
from openai import OpenAI

class ASDKnowledgeEnhancer:
    def __init__(self, asdkb_path: str, asd_texts_path: str, api_key: str = None, model_name: str = "qwen-plus", output_path: str = "data/enhanced_asd_texts.json"):
        """
        初始化ASD知识增强器

        Args:
            asdkb_path: 知识图谱文件路径
            asd_texts_path: 文本数据文件路径
            api_key: 大模型API密钥
            model_name: 模型名称
        """
        self.asdkb_path = asdkb_path
        self.asd_texts_path = asd_texts_path
        self.api_key = api_key
        self.model_name = model_name
        self.output_path = output_path

        # 存储知识图谱信息
        self.entities = {}  # 实体ID -> 实体信息
        self.entity_names = {}  # 实体名称 -> 实体ID列表
        self.relations = {}  # 关系信息
        self.synonyms = {}  # 同义词映射

        # 加载知识图谱和文本数据
        self.load_knowledge_base()
        self.load_text_data()

        print(f"知识图谱加载完成")
        print(f"   - 实体数量: {len(self.entities)}")
        print(f"   - 关系数量: {len(self.relations)}")
        print(f"   - 文本数量: {len(self.texts)}")

    def preprocess_pdf_text(self, raw_text: str) -> str:
        """
        使用大模型修正PDF提取的文本问题
        """
        correction_prompt = f"""
        你是一个专业的文本处理助手，擅长处理PDF提取文本的各种问题。请对从PDF中提取的文本进行以下处理：

        1. 纠正错别字：修正拼写错误和错别字
        2. 补全漏字：根据上下文语义补全缺失的字词
        3. 格式化清理：移除不必要的换行符、空格和格式混乱
        4. 语义连贯：确保修正后的文本语义通顺、逻辑连贯
        5. 专业术语保护：特别注意医学专业术语的准确性

        请保持原文的专业内容和核心含义不变，只进行必要的修正。

        待处理的文本：
        {raw_text}

        请直接输出修正后的完整文本，不要添加任何解释。
        """

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/",
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content":  correction_prompt},
                    {"role": "user", "content": f"请处理以下文本：{raw_text}"}
                ],
                temperature=0.1,  # 低温度确保确定性输出
                max_tokens=4000  # 根据文本长度调整
            )

            corrected_text = response.choices[0].message.content.strip()
            # print('原文本:',raw_text,'\n','修正文本',corrected_text)
            return corrected_text

        except Exception as e:
            print(f"文本修正失败: {e}")
            print(raw_text)
            return raw_text  # 失败时返回原文本

    def call_llm_api(self, text: str) -> List[Dict]:
        """
        调用大模型API提取实体
        Args:
            text: 输入文本
        Returns:
            提取到的实体列表
        """
        prompt = f"""
        角色：你是一位专业的AI助手，专门从事生物医学和临床文本的自然语言处理（NLP）和信息抽取（Information Extraction）。你的任务是精确识别并结构化输出文本中与自闭症谱系障碍（ASD）相关的实体。
        
        任务：请仔细分析用户提供的文本，识别并分类出以下四类与ASD相关的实体：
        干预方法（Intervention）：用于改善ASD核心症状、共病或功能的所有策略、疗法、项目和治疗方法。
        例如：基于前事的干预(ABI)、直接教学(DI)、回合式教学(DTT)、功能沟通训练(FCT)、功能行为分析(FBA)、任务分析(TA)、自我管理(SM)等。
        疾病与障碍（Disease/Disorder）：ASD本身、其共病以及其他相关的神经发育或精神障碍。
        例如：自闭症谱系障碍（ASD）、阿斯伯格综合征、雷特综合征、注意缺陷多动障碍（ADHD）、焦虑障碍、智力障碍、发育迟缓、癫痫、睡眠障碍、胃肠道问题等。
        症状与体征（Symptom/Sign）：ASD的核心行为特征及相关临床表现。
        例如：社交互动缺陷、语言沟通障碍、重复刻板行为、狭隘的兴趣、感觉处理异常（如对声音过敏）、眼神接触减少、模仿困难、情绪爆发、自伤行为、仪式化行为。
        筛查与评估工具（Screening/Assessment Tool）：用于识别、诊断或评估ASD症状和严重程度的标准化量表、问卷或工具。
        例如：修订版幼儿自闭症检查表(M-CHAT)、自闭症诊断观察量表（ADOS）、自闭症诊断访谈修订版（ADI-R）、儿童自闭症评定量表（CARS）、社会反应量表（SRS）、沟通与象征行为量表（CSBS）。
    
        输出要求：
        请以纯JSON数组格式输出结果，数组中的每个对象代表一个被识别的实体。
        每个JSON对象必须包含以下四个字段：
        "text": (字符串) 从原文中直接提取出的实体名称。
        "type": (字符串) 该实体的类别，必须是以下四种之一："干预方法", "疾病", "症状", "筛查量表"。
        "context": (字符串) 该实体在原文中出现的关键句子或短语片段，用于提供上下文。
        "confidence": (浮点数) 你对该实体识别结果的置信度，范围从0.0到1.0（1.0表示完全确定）。
        "start": (整数) 实体名称在原文中的起始位置。
        "end": (整数) 实体名称在原文中的结束位置。
        只输出JSON数组，不要有任何额外的解释、道歉或开场白。
        如果文本中没有找到任何相关实体，请输出一个空数组：[]。
        请确保实体的提取是准确且忠于原文的，避免编造或推断文中未明确提及的实体，格式如下：
        {{
          "entities": [
            {{
              "text": "实体文本",
              "type": "实体类型(干预方法/疾病/症状/筛查量表)",
              "context": "实体在原文中出现的关键句子或短语片段",
              "confidence": "对该实体识别结果的置信度",
              "start": 起始位置,
              "end": 结束位置
            }}
          ]
        }}"""
        try:
            client = OpenAI(
                # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            completion = client.chat.completions.create(
                model=self.model_name,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=[
                    {
                        "role": "system",  # 使用system角色
                         "content": prompt  # 详细的指令定义
                    },
                    {
                        "role": "user",
                        "content": f"请分析以下文本：{text}"  # 用户输入
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            content = completion.choices[0].message.content
            # print(completion.choices[0].message.content)
            # 解析JSON结果
            try:
                entity_data = json.loads(content)
                # 检查返回的是字典还是列表
                if isinstance(entity_data, dict):
                    # 如果是字典，按原逻辑处理
                    return entity_data.get('entities', [])
                elif isinstance(entity_data, list):
                    # 如果是列表，直接返回
                    return entity_data
                else:
                    print(f"未知的返回格式: {type(entity_data)}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始内容: {content}")
                return []

        except Exception as e:
            print(f"调用大模型API时出错: {str(e)}")
            traceback.print_exc()  # 打印详细错误信息
            return []

    def load_knowledge_base(self):
        """加载ASD知识图谱"""
        print("正在加载ASD知识图谱...")

        with open(self.asdkb_path, 'r', encoding='utf-8-sig') as f:
            kb_data = json.load(f)

        # 处理知识图谱数据
        for item in kb_data:
            if 'p' in item:
                relation_info = item['p']

                # 处理起始节点
                if 'start' in relation_info:
                    start_node = relation_info['start']
                    self._process_entity(start_node)

                # 处理结束节点
                if 'end' in relation_info:
                    end_node = relation_info['end']
                    self._process_entity(end_node)

                # 处理关系
                self._process_relation(relation_info)

    def _process_entity(self, node: Dict):
        """处理单个实体节点"""
        if 'properties' in node:
            props = node['properties']
            entity_id = props.get('id', str(node.get('identity', '')))

            # 存储实体信息
            # 干预方法-id:ID,英文名称,中文名称,介绍,:LABEL
            # 疾病-id:ID,中文名称,英文名称,同义词,snomed-ct编码（SCTID）,icd-10编码,介绍,发病群体,病因,:LABEL
            # 疾病icd-10-id:ID,中文名称,英文名称,同义词,icd-10编码,介绍,发病群体,病因,:LABEL
            # 症状-id:ID,症状ID,症状所属疾病的SCTID,ICD-10编码,父级症状ID,相似症状ID,英文描述,中文描述,分类,标签,:LABEL
            # 筛查量表-id:ID,英文,名称,介绍,适用年龄,花费时间,使用者,评分标准,提出者,来源,:LABEL,
            entity_info = {
                'id': entity_id,
                'labels': node.get('labels', []),
                'properties': props,
                'name': props.get('name', props.get('中文描述',props.get('英文名称', props.get('英文', '')))),
                'description': props.get('介绍', props.get('中文描述', props.get('英文描述', ''))),
                'category': props.get('分类', props.get('标签', '')),
                'english_name': props.get('英文名称', props.get('英文描述', '')),
                'icd_code': props.get('icd-10编码', props.get('ICD-10编码', '')),
                'synonyms': self._extract_synonyms(props)
            }

            self.entities[entity_id] = entity_info

            # 建立名称到ID的映射
            names_to_map = [
                               entity_info['name'],
                               entity_info['english_name'],
                               entity_info['description'][:20] if entity_info['description'] else ''
                           ] + entity_info['synonyms']

            # 多重索引机制，为每个实体创建多个检索入口
            # 用户输入"社交障碍"也能找到"社会交往障碍"实体
            # 支持同义词匹配
            # 支持部分描述匹配
            for name in names_to_map:
                if name and len(name.strip()) > 0:
                    clean_name = name.strip()
                    if clean_name not in self.entity_names:
                        self.entity_names[clean_name] = []
                    self.entity_names[clean_name].append(entity_id)

    def _extract_synonyms(self, props: Dict) -> List[str]:
        """提取同义词"""
        synonyms = []
        synonym_fields = ['同义词', 'synonyms']

        for field in synonym_fields:
            if field in props and props[field] != 'none':
                if isinstance(props[field], str):
                    synonyms.extend([s.strip() for s in props[field].split(',') if s.strip()])
                elif isinstance(props[field], list):
                    synonyms.extend(props[field])

        return synonyms

    def _process_relation(self, relation_info: Dict):
        """处理关系信息"""
        if 'start' in relation_info and 'end' in relation_info:
            start_id = relation_info['start']['properties'].get('id', str(relation_info['start'].get('identity', '')))
            end_id = relation_info['end']['properties'].get('id', str(relation_info['end'].get('identity', '')))

            relation_key = f"{start_id}-{end_id}"
            self.relations[relation_key] = {
                'start_id': start_id,
                'end_id': end_id,
                'start_entity': relation_info['start'],
                'end_entity': relation_info['end'],
                'relation_type': 'related_to'  # 可以根据需要扩展
            }

    def load_text_data(self):
        """加载文本数据"""
        print("正在加载文本数据...")

        with open(self.asd_texts_path, 'r', encoding='utf-8') as f:
            self.texts = json.load(f)

    def match_entity_to_kb(self, entity_text: str, entity_type: str = None, threshold: int = 80) -> Dict:
        """
        将LLM提取的实体与知识库中的实体进行匹配
        Args:
            entity_text: 实体文本
            entity_type: 实体类型
            threshold: 匹配阈值
        Returns:
            匹配到的知识库实体信息
        """
        best_match = None
        best_score = 0
        best_entity_id = None

        # 方法1: 精确匹配,完全相同的字符串匹配
        if entity_text in self.entity_names:
            entity_ids = self.entity_names[entity_text]
            best_entity_id = self._select_best_entity_by_type(entity_ids, entity_type)
            if best_entity_id:
                return {
                    'entity_id': best_entity_id,
                    'matched_text': entity_text,
                    'confidence': 1.0,
                    'method': 'exact'
                }

        # 方法2: 模糊匹配,处理拼写错误、简写、同义词
        for kb_entity_name, entity_ids in self.entity_names.items():
            if len(kb_entity_name) < 2:
                continue

            # 计算相似度
            score = fuzz.ratio(entity_text, kb_entity_name)
            if score > best_score and score >= threshold:
                candidate_entity_id = self._select_best_entity_by_type(entity_ids, entity_type)
                if candidate_entity_id:
                    best_score = score
                    best_match = kb_entity_name
                    best_entity_id = candidate_entity_id

        if best_entity_id:
            return {
                'entity_id': best_entity_id,
                'matched_text': best_match,
                'confidence': best_score / 100.0,
                'method': 'fuzzy'
            }

        # 方法3: 基于分词的部分匹配,处理词序不同但含义相同的表达
        entity_words = jieba.lcut(entity_text)
        for kb_entity_name, entity_ids in self.entity_names.items():
            kb_words = jieba.lcut(kb_entity_name)

            # 计算词汇重叠度
            common_words = set(entity_words) & set(kb_words)
            if len(common_words) > 0:
                overlap_score = len(common_words) / max(len(entity_words), len(kb_words))
                if overlap_score >= 0.6:  # 60%以上重叠
                    candidate_entity_id = self._select_best_entity_by_type(entity_ids, entity_type)
                    if candidate_entity_id:
                        if overlap_score * 100 > best_score:
                            best_score = overlap_score * 100
                            best_match = kb_entity_name
                            best_entity_id = candidate_entity_id

        if best_entity_id:
            return {
                'entity_id': best_entity_id,
                'matched_text': best_match,
                'confidence': best_score / 100.0,
                'method': 'jieba'
            }

        return None


    def _select_best_entity_by_type(self, entity_ids: List[str], entity_type: str = None) -> str:
        """根据实体类型选择最佳实体"""
        if len(entity_ids) == 1:
            return entity_ids[0]

        if not entity_type:
            return entity_ids[0]

        # 实体类型到标签的映射
        type_label_mapping = {
            '干预方法': ['实证有效的干预方法', '治疗方法'],
            '疾病': ['疾病', 'icd10疾病'],
            '症状': ['症状', 'Symptom'],
            '筛查量表': ['筛查量表', '量表', 'Scale']
        }

        target_labels = type_label_mapping.get(entity_type, [])

        # 优先选择标签匹配的实体
        for entity_id in entity_ids:
            if entity_id in self.entities:
                entity_labels = self.entities[entity_id].get('labels', [])
                if any(label in target_labels for label in entity_labels):
                    return entity_id

        # 如果没有标签匹配，返回第一个
        return entity_ids[0]

    def retrieve_entity_knowledge(self, entity_id: str) -> Dict:
        """检索实体的知识信息"""
        if entity_id not in self.entities:
            return {}

        entity = self.entities[entity_id]
        knowledge = {
            'entity': entity,
            'properties': entity['properties'],
            'related_entities': [],
            'relations': []
        }

        # 查找相关实体和关系
        for relation_key, relation in self.relations.items():
            if relation['start_id'] == entity_id:
                # 该实体作为起始点的关系
                related_entity_id = relation['end_id']
                if related_entity_id in self.entities:
                    knowledge['related_entities'].append(self.entities[related_entity_id])
                    knowledge['relations'].append(relation)

            elif relation['end_id'] == entity_id:
                # 该实体作为结束点的关系
                related_entity_id = relation['start_id']
                if related_entity_id in self.entities:
                    knowledge['related_entities'].append(self.entities[related_entity_id])
                    knowledge['relations'].append(relation)

        return knowledge

    def construct_knowledge_prompt(self, entity_id: str, max_length: int = 200) -> str:
        """构造知识提示"""
        knowledge = self.retrieve_entity_knowledge(entity_id)

        if not knowledge:
            return ""

        entity = knowledge['entity']
        prompt_parts = []

        # 基本信息
        if entity['description']:
            prompt_parts.append(f"定义：{entity['description'][:100]}")
        if entity['icd_code']:
            prompt_parts.append(f"ICD编码：{entity['icd_code']}")

        # 从属性中提取重要信息
        props = entity['properties']
        important_props = ['分类', '标签', '发病群体', '同义词', '病因', '症状表现', '评分标准', '适用年龄', '提出者']
        for prop in important_props:
            if prop in props and props[prop] != 'none' and str(props[prop]).strip():
                value = str(props[prop])[:50]  # 限制长度
                prompt_parts.append(f"{prop}：{value}")

        if knowledge['related_entities']:
            related_names = [rel['name'] for rel in knowledge['related_entities'][:3]]
            if related_names:
                prompt_parts.append(f"相关：{', '.join(related_names)}")


        # 组合prompt并限制长度
        full_prompt = "，".join(prompt_parts)
        if len(full_prompt) > max_length:
            full_prompt = full_prompt[:max_length] + "..."

        return full_prompt



    def select_best_from_group(self, entity_group: List[Dict]) -> Dict:
        """从重叠实体组中选择最佳实体"""
        if len(entity_group) == 1:
            return entity_group[0]

        best_entity = None
        best_score = -1

        for entity in entity_group:
            score = 0
            # 置信度权重 (40%)
            score += entity['confidence'] * 0.4
            # 长度权重 (30%)
            length_score = (entity['end'] - entity['start']) / 10
            score += min(length_score, 0.3) * 0.3
            # 方法权重 (30%)
            method_scores = {'exact': 0.3, 'jieba': 0.2, 'fuzzy': 0.1}
            score += method_scores.get(entity['method'], 0)

            if score > best_score:
                best_score = score
                best_entity = entity

        return best_entity


    def enhance_text(self, text: str) -> str | tuple[str | Any, str, int]:
        """
        使用大模型增强单个文本
        Args:
            text: 原始文本
        Returns:
            增强后的文本
        """

        # 使用大模型提取实体
        corrected_text = self.preprocess_pdf_text(text)
        llm_entities = self.call_llm_api(corrected_text)

        enhanced_text = corrected_text
        offset = 0

        for i, entity in enumerate(llm_entities):
            entity_text = entity.get('text', '')
            entity_type = entity.get('type', '')
            start_pos = entity.get('start', 0)
            end_pos = entity.get('end', len(entity_text))

            # 位置验证，llm生成的都有问题
            pos = corrected_text.find(entity_text)
            if pos != -1:
                start_pos = pos
                end_pos = pos + len(entity_text)
            else:
                continue
            # print('更新后的位置:(', start_pos,',', end_pos,')')
            # 调整位置考虑偏移
            adjusted_start = start_pos + offset
            adjusted_end = end_pos + offset

            # 知识库匹配
            kb_match = self.match_entity_to_kb(entity_text, entity_type)

            if kb_match:
                knowledge_prompt = self.construct_knowledge_prompt(kb_match['entity_id'])
                enhanced_part = f"{entity_text}（{knowledge_prompt}）" if knowledge_prompt else entity_text
            else:
                type_description_map = {
                    '干预方法': '孤独症谱系障碍的干预方法',
                    '疾病': '孤独症谱系障碍相关疾病',
                    '症状': '孤独症谱系障碍的症状',
                    '筛查量表': '孤独症谱系障碍的筛查量表'
                }
                description = type_description_map.get(entity_type, f'孤独症谱系障碍的{entity_type}')
                enhanced_part = f"{entity_text}（{description}）"

            # 计算长度差异
            original_length = adjusted_end - adjusted_start
            new_length = len(enhanced_part)
            length_diff = new_length - original_length

            # 替换文本
            enhanced_text = (
                    enhanced_text[:adjusted_start] +
                    enhanced_part +
                    enhanced_text[adjusted_end:]
            )

            # 更新偏移量
            offset += length_diff

        return enhanced_text, corrected_text, len(llm_entities)

    def enhance_all_texts(self) -> List[Dict]:
        """
        增强所有文本
        Args:
            output_path: 输出文件路径
            use_llm: 是否使用大模型进行实体提取
        Returns:
            增强后的文本列表
        """
        print("开始文本增强...")
        enhanced_texts = []

        for i, text_item in enumerate(self.texts):
            if i % 100 == 0:
                print(f"   处理进度: {i}/{len(self.texts)}")

            original_text = text_item['text']

            enhanced_text, corrected_text, entities_count = self.enhance_text(original_text)

            enhanced_item = text_item.copy()
            enhanced_item['original_text'] = original_text
            enhanced_item['corrected_text'] = corrected_text
            enhanced_item['enhanced_text'] = enhanced_text
            enhanced_item['enhancement_info'] = {
                'entities_found': entities_count,
                'length_increase': len(enhanced_text) - len(original_text)
            }
            enhanced_texts.append(enhanced_item)

            # 添加延时避免API限流
            if self.api_key and i % 10 == 0:
                time.sleep(1)


        print(f"文本增强完成，共处理 {len(enhanced_texts)} 个文本")

        if self.output_path:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_texts, f, ensure_ascii=False, indent=2)
            print(f"增强结果已保存到: {self.output_path}")

        return enhanced_texts

    def analyze_enhancement_results(self, enhanced_texts: List[Dict]):
        """分析增强结果"""
        print("\n增强结果分析:")
        total_texts = len(enhanced_texts)

        enhanced_count = sum(1 for item in enhanced_texts
                             if item['enhancement_info']['entities_found'] > 0)

        avg_entities = sum(item['enhancement_info']['entities_found']
                           for item in enhanced_texts) / total_texts

        avg_length_increase = sum(item['enhancement_info']['length_increase']
                                  for item in enhanced_texts) / total_texts

        print(f"   总文本数: {total_texts}")
        print(f"   被增强文本数: {enhanced_count} ({enhanced_count / total_texts * 100:.1f}%)")
        print(f"   平均实体识别数: {avg_entities:.2f}")
        print(f"   平均长度增加: {avg_length_increase:.1f} 字符")

        # 展示几个增强示例
        print(f"\n增强示例:")
        examples = [item for item in enhanced_texts
                    if item['enhancement_info']['entities_found'] > 0][:3]

        for i, example in enumerate(examples):
            print(f"\n示例 {i + 1}:")
            print(f"原文: {example['original_text'][:100]}...")
            print(f"修正: {example['corrected_text'][:100]}...")
            print(f"增强: {example['enhanced_text'][:200]}...")
            print(f"长度增加: {example['enhancement_info']['length_increase']}")


def main():
    """主函数"""
    print("ASD知识图谱文本增强（支持大模型）")
    print("=" * 50)

    # 设置API密钥 最好配置到环境变量里
    api_key = 'sk-975eb0f625a34a8591510177f19d6dfd'

    # 初始化增强器
    enhancer = ASDKnowledgeEnhancer(
        asdkb_path='data/asdkb.json',
        asd_texts_path='data/asd_texts.json',
        api_key=api_key,
        model_name='qwen-plus-2025-07-14',
        output_path='data/enhanced_asd_texts.json'
    )

    # 测试单个文本增强
    test_text = "常用于评定症状的量表包括家长使用的儿童孤独症行为检查量表autism behavior checklist ABC、社交反应量表social responsive ness scale SRS等"
    print(f"\n测试文本增强:")
    print(f"原文: {test_text}")


    enhanced_text, corrected_text, len = enhancer.enhance_text(test_text)
    print(f"LLM修正: {corrected_text}")
    print(f"LLM增强: {enhanced_text}")
    print(f"实体识别数: {len}")


    # 增强所有文本
    enhanced_texts = enhancer.enhance_all_texts()

    # 分析结果
    enhancer.analyze_enhancement_results(enhanced_texts)
    return enhancer, enhanced_texts






if __name__ == "__main__":
    # 运行主程序
    enhancer, enhanced_texts = main()

