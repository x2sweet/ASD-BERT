import PyPDF2
import os
import re
import jieba
from typing import List, Dict
import json

class PDFTextExtractor:
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        self.extracted_texts = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    # 修复：确保extract_text()返回字符串
                    page_text = page.extract_text()
                    if isinstance(page_text, tuple):
                        page_text = page_text[0] if page_text else ""
                    elif page_text is None:
                        page_text = ""
                    text += str(page_text)
        except Exception as e:
            print(f"提取PDF {pdf_path} 时出错: {e}")
            try:
                import fitz  
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                doc.close()
                print(f"使用PyMuPDF成功提取: {pdf_path}")
            except ImportError:
                print("失败")
            except Exception as e2:
                print(f"失败: {e2}")
        return text
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、（）【】《》""''\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences
    
    def extract_all_pdfs(self) -> List[Dict[str, str]]:
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        all_data = []
        
        for pdf_file in pdf_files:
            print(f"正在处理: {pdf_file}")
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                print(f"警告: {pdf_file} 未提取到文本内容")
                continue
            cleaned_text = self.clean_text(raw_text)
            sentences = self.split_into_sentences(cleaned_text)
            for sentence in sentences:
                if len(sentence) > 20: 
                    all_data.append({
                        'text': sentence,
                        'source': pdf_file,
                        'domain': 'ASD_medical'
                    })
            
            print(f"从 {pdf_file} 提取了 {len([s for s in sentences if len(s) > 20])} 条有效文本")
        
        print(f"总共提取了 {len(all_data)} 条文本数据")
        return all_data
    
    def save_to_json(self, data: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到: {output_path}")

if __name__ == "__main__":
    pdf_folder = r"data/ASD_paper"
    extractor = PDFTextExtractor(pdf_folder)
    extracted_data = extractor.extract_all_pdfs()
    extractor.save_to_json(extracted_data, "data/asd_texts.json")