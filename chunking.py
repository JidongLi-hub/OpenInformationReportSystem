"""
先进的RAG文档切分模块

采用LlamaIndex风格的分层切分策略：
1. Section (章节): 最大的语义单元，通常对应一级或二级标题
2. Parent (父块): 中等粒度，用于提供上下文给LLM
3. Child (子块): 细粒度，用于Embedding和精准检索
4. Summary (摘要): LLM生成的章节摘要，增强检索效果

工作流程：
- 检索时使用Child块做Embedding匹配
- 命中Child后，通过parent_id索引拿到Parent块给LLM
- 可选地，通过section_id索引拿到整个章节摘要

"""

import re
import uuid
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import os


class ChunkType(Enum):
    """块类型枚举"""
    SECTION = "section"      # 章节（最大粒度）
    PARENT = "parent"        # 父块（中粒度，供LLM阅读）
    CHILD = "child"          # 子块（细粒度，用于检索）
    SUMMARY = "summary"      # 摘要（LLM生成）


@dataclass
class Chunk:
    """统一的块数据结构"""
    id: str                              # 唯一标识符
    chunk_type: ChunkType                # 块类型
    text: str                            # 原始文本
    file_name: str                       # 来源文件名
    
    # 层级关系
    section_id: Optional[str] = None     # 所属章节ID
    parent_id: Optional[str] = None      # 父块ID (仅Child有)
    
    # 元信息
    heading: Optional[str] = None        # 标题
    heading_level: int = 0               # 标题层级 (1-6)
    position: int = 0                    # 在文档中的位置
    
    # 子块列表 (用于Parent)
    child_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """转换为字典，便于存储"""
        return {
            "id": self.id,
            "chunk_type": self.chunk_type.value,
            "text": self.text,
            "file_name": self.file_name,
            "section_id": self.section_id,
            "parent_id": self.parent_id,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "position": self.position,
            "child_ids": self.child_ids
        }


@dataclass
class Section:
    """解析后的文档章节"""
    heading: str                          # 标题文本
    heading_level: int                    # 标题层级
    content: str                          # 章节内容（不含子章节）
    raw_text: str                         # 完整原始文本
    start_pos: int                        # 起始位置
    end_pos: int                          # 结束位置
    subsections: List['Section'] = field(default_factory=list)


class MarkdownParser:
    """智能Markdown文档解析器"""
    
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def parse(self, markdown_text: str) -> List[Section]:
        """
        解析Markdown文档，返回层次化的Section列表
        
        Args:
            markdown_text: Markdown格式的文本
            
        Returns:
            Section列表，每个Section包含子章节
        """
        if not markdown_text or not markdown_text.strip():
            return []
        
        # 找到所有标题及其位置
        headings = []
        for match in self.HEADING_PATTERN.finditer(markdown_text):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()
            headings.append({
                'level': level,
                'title': title,
                'start': start,
                'content_start': match.end()
            })
        
        # 如果没有标题，整个文档作为一个section
        if not headings:
            return [Section(
                heading="Document",
                heading_level=0,
                content=markdown_text.strip(),
                raw_text=markdown_text.strip(),
                start_pos=0,
                end_pos=len(markdown_text)
            )]
        
        # 计算每个section的内容范围
        for i, h in enumerate(headings):
            if i < len(headings) - 1:
                h['end'] = headings[i + 1]['start']
            else:
                h['end'] = len(markdown_text)
            
            # 获取该section的内容（从标题后到下一个标题前）
            h['content'] = markdown_text[h['content_start']:h['end']].strip()
            h['raw_text'] = markdown_text[h['start']:h['end']].strip()
        
        # 构建层次结构
        sections = self._build_hierarchy(headings)
        
        return sections
    def parse_plain_text(self, text: str, title: str = "Document") -> List[Section]:
        """
        解析纯文本文档（无Markdown标记）
        
        将整个文档作为一个Section处理
        
        Args:
            text: 纯文本内容
            title: 文档标题（默认使用文件名）
            
        Returns:
            包含单个Section的列表
        """
        if not text or not text.strip():
            return []
        
        return [Section(
            heading=title,
            heading_level=1,
            content=text.strip(),
            raw_text=text.strip(),
            start_pos=0,
            end_pos=len(text)
        )]
    
    def _build_hierarchy(self, headings: List[dict]) -> List[Section]:
        """构建层次化的Section结构"""
        if not headings:
            return []
        
        root_sections = []
        stack = []  # [(level, section)]
        
        for h in headings:
            section = Section(
                heading=h['title'],
                heading_level=h['level'],
                content=h['content'],
                raw_text=h['raw_text'],
                start_pos=h['start'],
                end_pos=h['end']
            )
            
            # 找到合适的父节点
            while stack and stack[-1][0] >= h['level']:
                stack.pop()
            
            if stack:
                # 作为子节点添加
                stack[-1][1].subsections.append(section)
            else:
                # 作为根节点
                root_sections.append(section)
            
            stack.append((h['level'], section))
        
        return root_sections
    
    def flatten_sections(self, sections: List[Section], min_level: int = 1, max_level: int = 3) -> List[Section]:
        """
        将层次化的Section扁平化
        
        Args:
            sections: 层次化的Section列表
            min_level: 最小标题级别
            max_level: 最大标题级别（超过此级别合并到父节点）
        """
        result = []
        
        def _flatten(section: Section, accumulated_content: str = ""):
            # 将子章节内容合并
            full_content = section.content
            
            for sub in section.subsections:
                if sub.heading_level <= max_level:
                    # 独立处理
                    _flatten(sub, "")
                else:
                    # 合并到父节点
                    full_content += "\n\n" + sub.raw_text
            
            # 更新section内容并添加
            result.append(Section(
                heading=section.heading,
                heading_level=section.heading_level,
                content=full_content.strip(),
                raw_text=section.raw_text,
                start_pos=section.start_pos,
                end_pos=section.end_pos
            ))
        
        for section in sections:
            _flatten(section)
        
        return result


class HierarchicalChunker:
    """
    分层切分器
    
    实现父子块切分策略：
    - Section: 完整的章节（用于全局理解）
    - Parent: 中等粒度块，通常是一个段落或几个相关段落（给LLM阅读）
    - Child: 细粒度块，用于Embedding和检索
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 800,       # 父块目标大小（词数）
        child_chunk_size: int = 200,         # 子块目标大小（词数）
        parent_overlap: int = 100,           # 父块重叠
        child_overlap: int = 50,             # 子块重叠
        min_chunk_size: int = 30,            # 最小块大小
        section_level_threshold: int = 2,    # Section级别阈值（1-2级作为Section）
        max_section_words: int = 2000,       # Section最大词数（超过则截断/切分）
        max_embedding_chars: int = 20000,    # Embedding最大字符数（安全阈值）
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.min_chunk_size = min_chunk_size
        self.section_level_threshold = section_level_threshold
        self.max_section_words = max_section_words
        self.max_embedding_chars = max_embedding_chars
        
        self.parser = MarkdownParser()
    
    def chunk_document(self, markdown_text: str, file_name: str) -> Dict[str, List[Chunk]]:
        """
        对文档进行分层切分
        
        Args:
            markdown_text: Markdown格式文本
            file_name: 源文件名
            
        Returns:
            包含各层级块的字典：
            {
                'sections': [Section级别的Chunk],
                'parents': [Parent级别的Chunk],
                'children': [Child级别的Chunk，用于检索]
            }
        """
        # 解析文档结构
        sections = self.parser.parse(markdown_text)
        flattened = self.parser.flatten_sections(sections, max_level=self.section_level_threshold)
        
        result = {
            'sections': [],
            'parents': [],
            'children': []
        }
        
        position = 0
        
        for section in flattened:
            if not section.content.strip():
                continue
            
            # 检查Section是否过大，需要切分
            section_text = section.raw_text
            section_word_count = self._count_words(section_text)
            section_char_count = len(section_text)
            
            if section_word_count > self.max_section_words or section_char_count > self.max_embedding_chars:
                # Section过大，切分为多个子Section
                sub_section_chunks = self._split_large_section(
                    section, file_name, position
                )
                
                for sub_chunk in sub_section_chunks:
                    result['sections'].append(sub_chunk)
                    
                    # 将子Section内容切分为Parent块
                    parent_chunks = self._create_parent_chunks(
                        sub_chunk.text,  # 使用切分后的文本
                        sub_chunk.id,
                        file_name,
                        sub_chunk.heading,
                        position
                    )
                    
                    # 为每个Parent创建Child块
                    for parent in parent_chunks:
                        child_chunks = self._create_child_chunks(
                            parent.text,
                            parent.id,
                            sub_chunk.id,
                            file_name,
                            position
                        )
                        
                        parent.child_ids = [c.id for c in child_chunks]
                        
                        result['parents'].append(parent)
                        result['children'].extend(child_chunks)
                        
                        position += 1
                    
                    position += 1
            else:
                # Section大小合适，正常处理
                section_id = self._generate_id(section.raw_text, file_name, "section")
                section_chunk = Chunk(
                    id=section_id,
                    chunk_type=ChunkType.SECTION,
                    text=section.raw_text,
                    file_name=file_name,
                    heading=section.heading,
                    heading_level=section.heading_level,
                    position=position
                )
                result['sections'].append(section_chunk)
                
                # 将Section内容切分为Parent块
                parent_chunks = self._create_parent_chunks(
                    section.content,
                    section_id,
                    file_name,
                    section.heading,
                    position
                )
                
                # 为每个Parent创建Child块
                for parent in parent_chunks:
                    child_chunks = self._create_child_chunks(
                        parent.text,
                        parent.id,
                        section_id,
                        file_name,
                        position
                    )
                    
                    # 更新Parent的child_ids
                    parent.child_ids = [c.id for c in child_chunks]
                    
                    result['parents'].append(parent)
                    result['children'].extend(child_chunks)
                    
                    position += 1
                
                position += 1
        
        return result
    
    def chunk_plain_text(self, text: str, file_name: str, title: str = None) -> Dict[str, List[Chunk]]:
        """
        对纯文本文档进行分层切分
        
        将整个文档作为一个Section，然后切分为Parent和Child块
        
        Args:
            text: 纯文本内容
            file_name: 源文件名
            title: 文档标题（默认使用文件名，去掉扩展名）
            
        Returns:
            包含各层级块的字典：
            {
                'sections': [Section级别的Chunk],
                'parents': [Parent级别的Chunk],
                'children': [Child级别的Chunk，用于检索]
            }
        """
        if not text or not text.strip():
            return {'sections': [], 'parents': [], 'children': []}
        
        # 使用文件名作为默认标题
        if title is None:
            title = os.path.splitext(file_name)[0] if file_name else "Document"
        
        # 解析为单个Section
        sections = self.parser.parse_plain_text(text, title)
        
        result = {
            'sections': [],
            'parents': [],
            'children': []
        }
        
        position = 0
        
        for section in sections:
            if not section.content.strip():
                continue
            
            section_text = section.raw_text
            section_word_count = self._count_words(section_text)
            section_char_count = len(section_text)
            
            # 检查Section是否过大，需要切分
            if section_word_count > self.max_section_words or section_char_count > self.max_embedding_chars:
                # Section过大，切分为多个子Section
                sub_section_chunks = self._split_large_section(
                    section, file_name, position
                )
                
                for sub_chunk in sub_section_chunks:
                    result['sections'].append(sub_chunk)
                    
                    parent_chunks = self._create_parent_chunks(
                        sub_chunk.text,
                        sub_chunk.id,
                        file_name,
                        sub_chunk.heading,
                        position
                    )
                    
                    for parent in parent_chunks:
                        child_chunks = self._create_child_chunks(
                            parent.text,
                            parent.id,
                            sub_chunk.id,
                            file_name,
                            position
                        )
                        
                        parent.child_ids = [c.id for c in child_chunks]
                        
                        result['parents'].append(parent)
                        result['children'].extend(child_chunks)
                        
                        position += 1
                    
                    position += 1
            else:
                # Section大小合适，正常处理
                section_id = self._generate_id(section.raw_text, file_name, "section")
                section_chunk = Chunk(
                    id=section_id,
                    chunk_type=ChunkType.SECTION,
                    text=section.raw_text,
                    file_name=file_name,
                    heading=section.heading,
                    heading_level=section.heading_level,
                    position=position
                )
                result['sections'].append(section_chunk)
                
                parent_chunks = self._create_parent_chunks(
                    section.content,
                    section_id,
                    file_name,
                    section.heading,
                    position
                )
                
                for parent in parent_chunks:
                    child_chunks = self._create_child_chunks(
                        parent.text,
                        parent.id,
                        section_id,
                        file_name,
                        position
                    )
                    
                    parent.child_ids = [c.id for c in child_chunks]
                    
                    result['parents'].append(parent)
                    result['children'].extend(child_chunks)
                    
                    position += 1
                
                position += 1
        
        return result
    
    def chunk_auto(self, text: str, file_name: str) -> Dict[str, List[Chunk]]:
        """
        自动检测文档类型并进行切分
        
        根据文件扩展名和内容特征自动选择切分策略：
        - .md 文件或包含Markdown标题的文本：使用Markdown切分
        - .txt 文件或纯文本：使用纯文本切分
        
        Args:
            text: 文档内容
            file_name: 文件名
            
        Returns:
            包含各层级块的字典
        """
        # 检查文件扩展名
        ext = os.path.splitext(file_name)[1].lower() if file_name else ""
        
        # 检查是否包含Markdown标题
        has_markdown_headings = bool(self.parser.HEADING_PATTERN.search(text))
        
        if ext == '.md' or has_markdown_headings:
            # 使用Markdown切分
            return self.chunk_document(text, file_name)
        else:
            # 使用纯文本切分
            return self.chunk_plain_text(text, file_name)
    

    
    def _split_large_section(
        self, 
        section: Section, 
        file_name: str, 
        base_position: int
    ) -> List[Chunk]:
        """
        切分过大的Section为多个子Section块
        
        Args:
            section: 原始Section
            file_name: 文件名
            base_position: 基础位置
        
        Returns:
            切分后的Section Chunk列表
        """
        chunks = []
        text = section.raw_text
        heading = section.heading
        
        # 首先尝试按段落切分
        paragraphs = self._split_into_paragraphs(text)
        
        current_text = []
        current_word_count = 0
        part_index = 0
        
        for para in paragraphs:
            para_words = self._count_words(para)
            para_chars = len(para)
            
            # 如果单个段落就超限，需要强制切分
            if para_words > self.max_section_words or para_chars > self.max_embedding_chars:
                # 先保存当前累积的内容
                if current_text:
                    chunk_text = "\n\n".join(current_text)
                    chunk_heading = f"{heading} (Part {part_index + 1})" if part_index > 0 else heading
                    chunk_id = self._generate_id(chunk_text, file_name, f"section_p{part_index}")
                    
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type=ChunkType.SECTION,
                        text=chunk_text,
                        file_name=file_name,
                        heading=chunk_heading,
                        heading_level=section.heading_level,
                        position=base_position + part_index
                    ))
                    part_index += 1
                    current_text = []
                    current_word_count = 0
                
                # 强制切分大段落
                sub_texts = self._force_split_text(para, self.max_section_words, self.max_embedding_chars)
                for sub_text in sub_texts:
                    chunk_heading = f"{heading} (Part {part_index + 1})"
                    chunk_id = self._generate_id(sub_text, file_name, f"section_p{part_index}")
                    
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type=ChunkType.SECTION,
                        text=sub_text,
                        file_name=file_name,
                        heading=chunk_heading,
                        heading_level=section.heading_level,
                        position=base_position + part_index
                    ))
                    part_index += 1
            
            # 检查是否会超限
            elif (current_word_count + para_words > self.max_section_words or 
                  len("\n\n".join(current_text + [para])) > self.max_embedding_chars):
                # 保存当前块
                if current_text:
                    chunk_text = "\n\n".join(current_text)
                    chunk_heading = f"{heading} (Part {part_index + 1})" if part_index > 0 else heading
                    chunk_id = self._generate_id(chunk_text, file_name, f"section_p{part_index}")
                    
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type=ChunkType.SECTION,
                        text=chunk_text,
                        file_name=file_name,
                        heading=chunk_heading,
                        heading_level=section.heading_level,
                        position=base_position + part_index
                    ))
                    part_index += 1
                
                current_text = [para]
                current_word_count = para_words
            else:
                current_text.append(para)
                current_word_count += para_words
        
        # 处理剩余内容
        if current_text:
            chunk_text = "\n\n".join(current_text)
            chunk_heading = f"{heading} (Part {part_index + 1})" if part_index > 0 else heading
            chunk_id = self._generate_id(chunk_text, file_name, f"section_p{part_index}")
            
            chunks.append(Chunk(
                id=chunk_id,
                chunk_type=ChunkType.SECTION,
                text=chunk_text,
                file_name=file_name,
                heading=chunk_heading,
                heading_level=section.heading_level,
                position=base_position + part_index
            ))
        
        return chunks
    
    def _force_split_text(self, text: str, max_words: int, max_chars: int) -> List[str]:
        """
        强制切分过长文本，优先按句子边界，其次按字符
        
        Args:
            text: 需要切分的文本
            max_words: 最大词数
            max_chars: 最大字符数
        
        Returns:
            切分后的文本列表
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_text = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = self._count_words(sentence)
            sentence_chars = len(sentence)
            
            # 单个句子就超限，需要按字符强制切分
            if sentence_chars > max_chars:
                # 先保存当前内容
                if current_text:
                    chunks.append(" ".join(current_text))
                    current_text = []
                    current_word_count = 0
                
                # 按字符切分长句
                for i in range(0, len(sentence), max_chars - 100):  # 留余量
                    chunk = sentence[i:i + max_chars - 100]
                    chunks.append(chunk)
            
            elif (current_word_count + sentence_words > max_words or 
                  len(" ".join(current_text + [sentence])) > max_chars):
                if current_text:
                    chunks.append(" ".join(current_text))
                current_text = [sentence]
                current_word_count = sentence_words
            else:
                current_text.append(sentence)
                current_word_count += sentence_words
        
        if current_text:
            chunks.append(" ".join(current_text))
        
        return chunks if chunks else [text[:max_chars]]
    
    def _create_parent_chunks(
        self,
        text: str,
        section_id: str,
        file_name: str,
        heading: str,
        base_position: int
    ) -> List[Chunk]:
        """创建Parent级别的块"""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        current_text = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = self._count_words(para)
            
            # 如果单个段落太大，需要进一步切分
            if para_words > self.parent_chunk_size:
                # 先保存当前累积的内容
                if current_text:
                    chunk_text = "\n\n".join(current_text)
                    chunk_id = self._generate_id(chunk_text, file_name, "parent")
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type=ChunkType.PARENT,
                        text=chunk_text,
                        file_name=file_name,
                        section_id=section_id,
                        heading=heading,
                        position=base_position + len(chunks)
                    ))
                    current_text = []
                    current_word_count = 0
                
                # 切分大段落
                sub_chunks = self._split_large_text(para, self.parent_chunk_size, self.parent_overlap)
                for sub in sub_chunks:
                    chunk_id = self._generate_id(sub, file_name, "parent")
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type=ChunkType.PARENT,
                        text=sub,
                        file_name=file_name,
                        section_id=section_id,
                        heading=heading,
                        position=base_position + len(chunks)
                    ))
            
            elif current_word_count + para_words > self.parent_chunk_size:
                # 当前块已满，保存并开始新块
                if current_text:
                    chunk_text = "\n\n".join(current_text)
                    chunk_id = self._generate_id(chunk_text, file_name, "parent")
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type=ChunkType.PARENT,
                        text=chunk_text,
                        file_name=file_name,
                        section_id=section_id,
                        heading=heading,
                        position=base_position + len(chunks)
                    ))
                    
                    # 重叠：保留最后一段
                    current_text = [current_text[-1]] if current_text else []
                    current_word_count = self._count_words(current_text[0]) if current_text else 0
                
                current_text.append(para)
                current_word_count += para_words
            
            else:
                current_text.append(para)
                current_word_count += para_words
        
        # 处理剩余内容
        if current_text:
            chunk_text = "\n\n".join(current_text)
            if self._count_words(chunk_text) >= self.min_chunk_size:
                chunk_id = self._generate_id(chunk_text, file_name, "parent")
                chunks.append(Chunk(
                    id=chunk_id,
                    chunk_type=ChunkType.PARENT,
                    text=chunk_text,
                    file_name=file_name,
                    section_id=section_id,
                    heading=heading,
                    position=base_position + len(chunks)
                ))
            elif chunks:
                # 太短则合并到上一个块
                chunks[-1].text += "\n\n" + chunk_text
        
        # 如果没有生成任何块，至少创建一个
        if not chunks and text.strip():
            chunk_id = self._generate_id(text, file_name, "parent")
            chunks.append(Chunk(
                id=chunk_id,
                chunk_type=ChunkType.PARENT,
                text=text.strip(),
                file_name=file_name,
                section_id=section_id,
                heading=heading,
                position=base_position
            ))
        
        return chunks
    
    def _create_child_chunks(
        self,
        parent_text: str,
        parent_id: str,
        section_id: str,
        file_name: str,
        base_position: int
    ) -> List[Chunk]:
        """创建Child级别的块（用于检索）"""
        chunks = []
        
        # 按句子边界切分
        sentences = self._split_into_sentences(parent_text)
        
        current_text = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = self._count_words(sentence)
            
            if current_word_count + sentence_words > self.child_chunk_size and current_text:
                # 保存当前块
                chunk_text = " ".join(current_text)
                chunk_id = self._generate_id(chunk_text, file_name, "child")
                chunks.append(Chunk(
                    id=chunk_id,
                    chunk_type=ChunkType.CHILD,
                    text=chunk_text,
                    file_name=file_name,
                    section_id=section_id,
                    parent_id=parent_id,
                    position=base_position + len(chunks)
                ))
                
                # 重叠
                overlap_words = 0
                overlap_sentences = []
                for s in reversed(current_text):
                    s_words = self._count_words(s)
                    if overlap_words + s_words <= self.child_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break
                
                current_text = overlap_sentences
                current_word_count = overlap_words
            
            current_text.append(sentence)
            current_word_count += sentence_words
        
        # 处理剩余内容
        if current_text:
            chunk_text = " ".join(current_text)
            if self._count_words(chunk_text) >= self.min_chunk_size:
                chunk_id = self._generate_id(chunk_text, file_name, "child")
                chunks.append(Chunk(
                    id=chunk_id,
                    chunk_type=ChunkType.CHILD,
                    text=chunk_text,
                    file_name=file_name,
                    section_id=section_id,
                    parent_id=parent_id,
                    position=base_position + len(chunks)
                ))
            elif chunks:
                # 合并到上一个块
                chunks[-1].text += " " + chunk_text
        
        # 确保至少有一个child块
        if not chunks and parent_text.strip():
            chunk_id = self._generate_id(parent_text, file_name, "child")
            chunks.append(Chunk(
                id=chunk_id,
                chunk_type=ChunkType.CHILD,
                text=parent_text.strip(),
                file_name=file_name,
                section_id=section_id,
                parent_id=parent_id,
                position=base_position
            ))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """按段落分割"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """按句子边界分割"""
        # 支持英文和中文句子边界
        sentence_endings = r'(?<=[.!?。！？])\s+'
        sentences = re.split(sentence_endings, text)
        
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)
        
        return result if result else [text.strip()]
    
    def _split_large_text(self, text: str, max_words: int, overlap: int) -> List[str]:
        """切分过大的文本"""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            start = end - overlap if overlap > 0 and end < len(words) else end
        
        return chunks
    
    def _count_words(self, text: str) -> int:
        """统计词数（同时支持中英文）"""
        # 英文单词
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        # 中文字符（每个字符算一个词）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return english_words + chinese_chars
    
    def _generate_id(self, text: str, file_name: str, prefix: str) -> str:
        """生成唯一ID"""
        content = f"{file_name}:{prefix}:{text[:100]}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_digest}"


def chunk_plain_text_document(
    text: str,
    file_name: str,
    title: str = None,
    parent_chunk_size: int = 800,
    child_chunk_size: int = 200,
    generate_summaries: bool = False,
    llm_base_url: str = "http://localhost:8888/v1"
) -> Dict[str, List[Chunk]]:
    """
    对纯文本文档进行分层切分
    
    Args:
        text: 纯文本内容
        file_name: 文件名
        title: 文档标题（可选）
        parent_chunk_size: 父块大小
        child_chunk_size: 子块大小
        generate_summaries: 是否生成摘要
        llm_base_url: LLM服务地址
    
    Returns:
        包含sections, parents, children, summaries的字典
    """
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size
    )
    
    result = chunker.chunk_plain_text(text, file_name, title)
    result['summaries'] = []
    
    if generate_summaries:
        try:
            summary_gen = SummaryGenerator(base_url=llm_base_url)
            for section in result['sections']:
                summary = summary_gen.generate_section_summary(section)
                if summary:
                    result['summaries'].append(summary)
        except Exception as e:
            print(f"[WARNING] 摘要生成初始化失败: {e}")
    
    return result


def chunk_document_auto(
    text: str,
    file_name: str,
    parent_chunk_size: int = 800,
    child_chunk_size: int = 200,
    generate_summaries: bool = False,
    llm_base_url: str = "http://localhost:8888/v1"
) -> Dict[str, List[Chunk]]:
    """
    自动检测文档类型并进行分层切分
    
    Args:
        text: 文档内容
        file_name: 文件名
        parent_chunk_size: 父块大小
        child_chunk_size: 子块大小
        generate_summaries: 是否生成摘要
        llm_base_url: LLM服务地址
    
    Returns:
        包含sections, parents, children, summaries的字典
    """
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size
    )
    
    result = chunker.chunk_auto(text, file_name)
    result['summaries'] = []
    
    if generate_summaries:
        try:
            summary_gen = SummaryGenerator(base_url=llm_base_url)
            for section in result['sections']:
                summary = summary_gen.generate_section_summary(section)
                if summary:
                    result['summaries'].append(summary)
        except Exception as e:
            print(f"[WARNING] 摘要生成初始化失败: {e}")
    
    return result


class SummaryGenerator:
    """
    使用LLM生成摘要
    
    为Section和Parent块生成摘要，增强检索效果
    """
    
    def __init__(self, base_url: str = "http://localhost:28888/v1", api_key: str = "EMPTY"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = None
        self._check_and_get_model()
    
    def _check_and_get_model(self):
        """检查LLM服务并获取模型名"""
        try:
            models = self.client.models.list()
            if models.data:
                self.model = models.data[0].id
                print(f"[INFO] 成功连接到LLM服务，使用模型: {self.model}")
        except Exception as e:
            print(f"[WARNING] 无法连接到LLM服务: {e}")
            print("[WARNING] 摘要生成功能将被禁用")
            self.model = None
    
    def generate_summary(self, text: str, context: str = "") -> Optional[str]:
        """
        生成文本摘要
        
        Args:
            text: 需要摘要的文本
            context: 额外的上下文信息（如章节标题）
        """
        if not self.model:
            return None
        
        prompt = f"""请为以下文档片段生成一个简洁的摘要（2-3句话），突出核心主题和关键信息。

上下文: {context}

文档内容:
{text[:3000]}

摘要:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文档摘要助手。生成简洁、准确的摘要。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARNING] 摘要生成失败: {e}")
            return None
    
    def generate_section_summary(self, section_chunk: Chunk) -> Optional[Chunk]:
        """为Section生成摘要块"""
        summary_text = self.generate_summary(
            section_chunk.text,
            context=f"章节标题: {section_chunk.heading}"
        )
        
        if summary_text:
            return Chunk(
                id=f"summary_{section_chunk.id}",
                chunk_type=ChunkType.SUMMARY,
                text=summary_text,
                file_name=section_chunk.file_name,
                section_id=section_chunk.id,
                heading=section_chunk.heading,
                heading_level=section_chunk.heading_level,
                position=section_chunk.position
            )
        return None


# 导出便捷函数
def chunk_markdown_document(
    markdown_text: str,
    file_name: str,
    parent_chunk_size: int = 800,
    child_chunk_size: int = 200,
    generate_summaries: bool = False,
    llm_base_url: str = "http://localhost:8888/v1"
) -> Dict[str, List[Chunk]]:
    """
    对Markdown文档进行分层切分
    
    Args:
        markdown_text: Markdown文本
        file_name: 文件名
        parent_chunk_size: 父块大小
        child_chunk_size: 子块大小
        generate_summaries: 是否生成摘要
        llm_base_url: LLM服务地址
    
    Returns:
        包含sections, parents, children, summaries的字典
    """
    chunker = HierarchicalChunker(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size
    )
    
    result = chunker.chunk_document(markdown_text, file_name)
    result['summaries'] = []
    
    if generate_summaries:
        try:
            summary_gen = SummaryGenerator(base_url=llm_base_url)
            for section in result['sections']:
                summary = summary_gen.generate_section_summary(section)
                if summary:
                    result['summaries'].append(summary)
        except Exception as e:
            print(f"[WARNING] 摘要生成初始化失败: {e}")
    
    return result


if __name__ == "__main__":
    # 测试代码
    test_md = """
# Foreign Relations of the United States

## Chapter 1: Introduction

This document provides an overview of the foreign relations policies.
The State Department plays a crucial role in diplomatic affairs.

International cooperation has been a cornerstone of American foreign policy.
Through various treaties and agreements, the United States has maintained
strong relationships with allied nations.

### Section 1.1: Historical Background

The history of American diplomacy dates back to the founding of the nation.
Early diplomatic efforts focused on establishing recognition from European powers.

### Section 1.2: Modern Era

Contemporary diplomacy involves complex multilateral negotiations.
Trade agreements, security pacts, and environmental treaties are common.

## Chapter 2: Key Agreements

This chapter discusses major international agreements signed by the United States.
"""
    
    result = chunk_markdown_document(test_md, "test.md")
    
    print("=== Sections ===")
    for s in result['sections']:
        print(f"ID: {s.id}, Heading: {s.heading}")
    
    print("\n=== Parents ===")
    for p in result['parents']:
        print(f"ID: {p.id}, Section: {p.section_id}, Words: {len(p.text.split())}")
    
    print("\n=== Children ===")
    for c in result['children']:
        print(f"ID: {c.id}, Parent: {c.parent_id}, Text: {c.text[:50]}...")
