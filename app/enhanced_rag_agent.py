import os
import json
import logging
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback if pydantic import fails
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(**kwargs):
        return None


try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunk(BaseModel):
    """Model for document chunks with metadata"""

    content: str = Field(description="The actual text content of the chunk")
    filename: str = Field(description="Source filename")
    chunk_id: int = Field(description="Chunk index within the document")
    start_char: int = Field(
        description="Starting character position in original document"
    )
    end_char: int = Field(description="Ending character position in original document")
    embedding: Optional[List[float]] = Field(
        default=None, description="OpenAI embedding vector"
    )


class RetrievalResult(BaseModel):
    """Model for retrieval results with scores"""

    chunk: DocumentChunk
    similarity_score: float = Field(description="Cosine similarity score")
    rerank_score: float = Field(description="Re-ranking score")
    final_score: float = Field(description="Final combined score")


class RelevanceCheck(BaseModel):
    """Model for relevance checking response"""

    relevant: bool = Field(
        description="Whether the query is relevant to the legal documents"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the relevance decision")


class TextSplitter:
    """Simple text splitter with overlap - lightweight alternative to LangChain"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, filename: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # Extract chunk content
            chunk_content = text[start:end].strip()

            if chunk_content:  # Only add non-empty chunks
                chunk = DocumentChunk(
                    content=chunk_content,
                    filename=filename,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)
                chunk_id += 1

            # Move start position with overlap
            if end == len(text):
                break
            start = end - self.chunk_overlap

        return chunks


class EnhancedRAGAgent:
    def __init__(
        self, chunk_size: int = 4000, chunk_overlap: int = 150, top_k: int = 5
    ):
        logger.info("🚀 Initializing Enhanced RAG Agent with OpenAI Embeddings...")

        # Check if OpenAI is available
        if openai is None:
            logger.error("❌ OpenAI package not available")
            raise ImportError("OpenAI package is required but not installed")

        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY is required")

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialized successfully")

        # Initialize text splitter
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.top_k = top_k

        # Load and process documents
        self.rules_content = self._load_rules()
        self.document_chunks = self._load_and_chunk_documents()

        # Initialize or load embeddings
        self._initialize_embeddings()

        logger.info("🎉 Enhanced RAG Agent initialization complete!")

    def _load_rules(self) -> str:
        """Load rules.txt content"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        rules_path = os.path.join(project_root, "raw_data", "rules.txt")

        logger.info(f"📁 Loading rules from: {rules_path}")

        try:
            if not os.path.exists(rules_path):
                logger.warning(f"⚠️ Rules file not found at: {rules_path}")
                alt_rules_path = os.path.join(os.getcwd(), "raw_data", "rules.txt")
                if os.path.exists(alt_rules_path):
                    rules_path = alt_rules_path
                else:
                    return "Thông tin về các tài liệu pháp lý không khả dụng."

            with open(rules_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(
                    f"✅ Rules file loaded successfully, {len(content)} characters"
                )
                return content
        except Exception as e:
            logger.error(f"❌ Error loading rules file: {e}")
            return "Có lỗi xảy ra khi tải thông tin tài liệu."

    def _load_and_chunk_documents(self) -> List[DocumentChunk]:
        """Load all documents and split them into chunks"""
        all_chunks = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, "data")

        logger.info(f"📂 Loading documents from: {data_path}")

        try:
            if not os.path.exists(data_path):
                alt_data_path = os.path.join(os.getcwd(), "data")
                if os.path.exists(alt_data_path):
                    data_path = alt_data_path
                else:
                    return []

            md_files = [f for f in os.listdir(data_path) if f.endswith(".md")]
            logger.info(f"📄 Found {len(md_files)} .md files")

            for filename in md_files:
                file_path = os.path.join(data_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Split document into chunks
                    chunks = self.text_splitter.split_text(content, filename)
                    all_chunks.extend(chunks)

                    logger.info(
                        f"📖 Processed {filename}: {len(content)} chars → {len(chunks)} chunks"
                    )

                except Exception as e:
                    logger.error(f"❌ Error reading {filename}: {e}")

            logger.info(f"✅ Total chunks created: {len(all_chunks)}")
            return all_chunks

        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
            return []

    def _get_cache_path(self) -> str:
        """Get cache file path for embeddings"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, ".cache")

        # Check if we're in a read-only environment (like Vercel)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(cache_dir, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return os.path.join(cache_dir, "embeddings_cache.pkl")
        except (OSError, PermissionError):
            # Read-only file system - return None to disable caching
            logger.warning(
                "⚠️ Read-only file system detected - disabling embedding cache"
            )
            return None

    def _calculate_content_hash(self) -> str:
        """Calculate hash of all document contents for cache validation"""
        content_str = ""
        for chunk in self.document_chunks:
            content_str += f"{chunk.filename}:{chunk.content}"
        return hashlib.md5(content_str.encode()).hexdigest()

    def _initialize_embeddings(self):
        """Initialize or load cached embeddings"""
        cache_path = self._get_cache_path()

        # If cache_path is None, we're in a read-only environment
        if cache_path is None:
            logger.info(
                "🔄 Generating embeddings without caching (read-only environment)..."
            )
            self._generate_embeddings(None, None)
            return

        current_hash = self._calculate_content_hash()

        # Try to load from cache
        if os.path.exists(cache_path):
            try:
                logger.info("🔄 Loading embeddings from cache...")
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                if cache_data.get("content_hash") == current_hash:
                    logger.info("✅ Cache is valid, loading embeddings")
                    embeddings = cache_data["embeddings"]

                    # Assign embeddings to chunks
                    for i, chunk in enumerate(self.document_chunks):
                        if i < len(embeddings):
                            chunk.embedding = embeddings[i]

                    logger.info(f"✅ Loaded {len(embeddings)} embeddings from cache")
                    return
                else:
                    logger.info("⚠️ Cache is outdated, will regenerate embeddings")
            except Exception as e:
                logger.warning(f"⚠️ Error loading cache: {e}")

        # Generate new embeddings
        logger.info("🔄 Generating new embeddings...")
        self._generate_embeddings(cache_path, current_hash)

    def _generate_embeddings(
        self, cache_path: Optional[str], content_hash: Optional[str]
    ):
        """Generate embeddings for all chunks"""
        texts = [chunk.content for chunk in self.document_chunks]

        if not texts:
            logger.warning("⚠️ No text content to embed")
            return

        logger.info(f"🧠 Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings in batches to avoid API limits
        # Use smaller batch size for Vercel to reduce initialization time
        batch_size = 50
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"📡 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}"
            )

            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",  # More cost-effective model
                    input=batch,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"❌ Error generating embeddings for batch: {e}")
                # Use zero vectors as fallback
                fallback_embeddings = [[0.0] * 1536] * len(batch)
                all_embeddings.extend(fallback_embeddings)

        # Assign embeddings to chunks
        for i, chunk in enumerate(self.document_chunks):
            if i < len(all_embeddings):
                chunk.embedding = all_embeddings[i]

        # Cache the embeddings (only if cache_path is available)
        if cache_path is not None and content_hash is not None:
            try:
                cache_data = {
                    "content_hash": content_hash,
                    "embeddings": all_embeddings,
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"💾 Cached {len(all_embeddings)} embeddings")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache embeddings: {e}")
        else:
            logger.info(
                f"✅ Generated {len(all_embeddings)} embeddings (no caching in read-only environment)"
            )

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small", input=[query]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Error getting query embedding: {e}")
            return [0.0] * 1536  # Fallback zero vector

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays for efficient computation
            a = np.array(vec1)
            b = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return dot_product / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"❌ Error calculating cosine similarity: {e}")
            return 0.0

    def _rerank_chunks(
        self, query: str, initial_results: List[Tuple[DocumentChunk, float]]
    ) -> List[RetrievalResult]:
        """Re-rank chunks using multiple factors"""
        reranked_results = []

        # Enhanced keyword-based re-ranking
        query_words = set(query.lower().split())

        # Extract phone numbers from query if present
        import re

        phone_numbers = re.findall(r"\d{4}\.\d{3}\.\d{3}|\d{10}", query)

        for chunk, similarity_score in initial_results:
            # Calculate keyword overlap score
            chunk_words = set(chunk.content.lower().split())
            keyword_overlap = len(query_words.intersection(chunk_words)) / max(
                len(query_words), 1
            )

            # Enhanced phone number matching
            phone_bonus = 0.0
            if phone_numbers:
                for phone in phone_numbers:
                    if phone in chunk.content or phone.replace(
                        ".", ""
                    ) in chunk.content.replace(".", ""):
                        phone_bonus = 0.5  # High bonus for exact phone number match

            # Calculate position bonus (earlier chunks might be more important)
            position_bonus = 1.0 / (chunk.chunk_id + 1) * 0.05

            # Calculate filename relevance (some files might be more relevant)
            filename_bonus = (
                0.1
                if any(word in chunk.filename.lower() for word in query_words)
                else 0.0
            )

            # Content length bonus for more comprehensive chunks
            content_length_bonus = min(len(chunk.content) / 4000.0, 0.1)

            # Combine scores with enhanced weighting
            rerank_score = (
                similarity_score * 0.6  # Embedding similarity (primary)
                + keyword_overlap * 0.2  # Keyword overlap
                + phone_bonus  # Phone number exact match (highest priority)
                + position_bonus  # Position in document
                + filename_bonus  # Filename relevance
                + content_length_bonus  # Prefer more comprehensive chunks
            )

            final_score = (
                similarity_score * 0.4 + rerank_score * 0.6
            )  # Favor re-ranking more

            result = RetrievalResult(
                chunk=chunk,
                similarity_score=similarity_score,
                rerank_score=rerank_score,
                final_score=final_score,
            )
            reranked_results.append(result)

        # Sort by final score
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(f"🔄 Re-ranked {len(reranked_results)} results")

        # Log phone number matches for debugging
        if phone_numbers:
            logger.info(f"📞 Looking for phone numbers: {phone_numbers}")
            for i, result in enumerate(reranked_results[:3]):
                has_phone = any(
                    phone in result.chunk.content for phone in phone_numbers
                )
                logger.info(
                    f"📄 #{i+1}: {result.chunk.filename} (final: {result.final_score:.3f}, has_phone: {has_phone})"
                )

        return reranked_results

    def _retrieve_relevant_chunks(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Retrieve and re-rank relevant document chunks"""
        if top_k is None:
            top_k = self.top_k

        logger.info(f"🔍 Retrieving relevant chunks for query: {query[:100]}...")

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Calculate similarities
        similarities = []
        for chunk in self.document_chunks:
            if chunk.embedding is None:
                continue

            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            similarities.append((chunk, similarity))

        # Sort by similarity and take top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Get more candidates for better re-ranking (especially for complex documents)
        top_candidates = similarities[: top_k * 4]  # Increased from 2x to 4x

        logger.info(f"📊 Found {len(top_candidates)} candidate chunks")

        # Re-rank the candidates
        reranked_results = self._rerank_chunks(query, top_candidates)

        # Enhanced result selection: ensure we get comprehensive coverage
        final_results = self._select_comprehensive_results(
            reranked_results, top_k, query
        )

        logger.info(f"✅ Retrieved {len(final_results)} relevant chunks")
        for i, result in enumerate(final_results[:5]):  # Log top 5
            logger.info(
                f"📄 #{i+1}: {result.chunk.filename} chunk_{result.chunk.chunk_id} (sim: {result.similarity_score:.3f}, final: {result.final_score:.3f})"
            )

        return final_results

    def _select_comprehensive_results(
        self, reranked_results: List[RetrievalResult], top_k: int, query: str
    ) -> List[RetrievalResult]:
        """Select results ensuring comprehensive coverage from relevant documents"""
        selected_results = []
        document_chunk_count = {}

        # First pass: select top results and track document coverage
        for result in reranked_results:
            if (
                len(selected_results) >= top_k * 2
            ):  # Allow up to 2x normal results for comprehensive coverage
                break

            filename = result.chunk.filename
            document_chunk_count[filename] = document_chunk_count.get(filename, 0) + 1

            # Always include high-scoring results
            if result.final_score > 0.5 or len(selected_results) < top_k:
                selected_results.append(result)
            # Include additional chunks from the same document if they contain relevant info
            elif document_chunk_count[filename] <= 3 and result.final_score > 0.3:
                # Check if this chunk contains complementary information
                if self._has_complementary_info(result.chunk.content, query):
                    selected_results.append(result)

        # Sort final results by score
        selected_results.sort(key=lambda x: x.final_score, reverse=True)

        # Return top results but ensure we have comprehensive coverage
        return selected_results[
            : top_k * 2
        ]  # Allow more results for comprehensive answers

    def _has_complementary_info(self, content: str, query: str) -> bool:
        """Check if content has complementary information to the query"""
        import re

        # Check for structured data patterns (tables, lists, etc.)
        has_table = "|" in content or "Số TT" in content
        has_contact_info = bool(
            re.search(r"\d{4}\.\d{3}\.\d{3}|email|địa chỉ", content, re.IGNORECASE)
        )
        has_organizational_info = bool(
            re.search(r"phòng|khu vực|viện|đơn vị", content, re.IGNORECASE)
        )

        return has_table or has_contact_info or has_organizational_info

    def _check_relevance(self, question: str) -> RelevanceCheck:
        """Check if question is relevant using OpenAI"""
        logger.info(f"🤔 Checking relevance for question: {question[:100]}...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a legal assistant. Determine if the user's question is relevant to Vietnamese prosecution office documents.

Available documents:
{self.rules_content}

You must respond with ONLY valid JSON. No markdown, no code blocks, no explanation text.

Format:
{{"relevant": true, "confidence": 0.9, "reasoning": "brief explanation"}}

Examples:
{{"relevant": true, "confidence": 0.8, "reasoning": "Question asks about prosecution office addresses"}}
{{"relevant": false, "confidence": 0.9, "reasoning": "Question is about weather, not legal matters"}}""",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()
            logger.info(f"🔍 Relevance response: {result_text}")

            # Clean up markdown code blocks if present
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                result_text = "\n".join(json_lines).strip()

            result = json.loads(result_text)
            relevance_check = RelevanceCheck(**result)

            logger.info(
                f"✅ Relevance: {relevance_check.relevant}, confidence: {relevance_check.confidence}"
            )
            return relevance_check

        except Exception as e:
            logger.error(f"❌ Error in relevance check: {e}")
            return RelevanceCheck(
                relevant=True,
                confidence=0.5,
                reasoning="Error in relevance check, defaulting to relevant",
            )

    def get_streaming_response(
        self, question: str, conversation_history: List[Dict[str, str]] = None
    ):
        """Get streaming response using enhanced RAG with embeddings and re-ranking"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"🚀 NEW ENHANCED REQUEST at {timestamp}: {question}")

        if conversation_history is None:
            conversation_history = []

        # Step 1: Check relevance
        relevance_check = self._check_relevance(question)

        if not relevance_check.relevant:
            logger.info("🚫 Question not relevant - providing general response")
            return self._get_basic_response(question)

        # Step 2: Retrieve relevant chunks using embeddings + re-ranking
        relevant_results = self._retrieve_relevant_chunks(question)

        if not relevant_results:
            logger.warning("⚠️ No relevant chunks found")
            return self._get_no_docs_response(question)

        # Log retrieval statistics
        logger.info(
            f"📊 Retrieved {len(relevant_results)} chunks from {len(set(r.chunk.filename for r in relevant_results))} documents"
        )
        doc_stats = {}
        for result in relevant_results:
            filename = result.chunk.filename
            doc_stats[filename] = doc_stats.get(filename, 0) + 1

        for filename, count in doc_stats.items():
            logger.info(f"📄 {filename}: {count} chunks")

        # Step 3: Prepare context from relevant chunks
        context_parts = []
        used_files = set()

        # Group chunks by document for better organization
        chunks_by_document = {}
        for result in relevant_results:
            filename = result.chunk.filename
            if filename not in chunks_by_document:
                chunks_by_document[filename] = []
            chunks_by_document[filename].append(result)

        # Sort documents by highest scoring chunk
        sorted_documents = sorted(
            chunks_by_document.items(),
            key=lambda x: max(r.final_score for r in x[1]),
            reverse=True,
        )

        for filename, chunks in sorted_documents:
            # Add document header
            max_score = max(r.final_score for r in chunks)
            context_parts.append(
                f"**Document: {filename}** (Max Score: {max_score:.3f})"
            )

            # Sort chunks within document by chunk_id to maintain logical order
            chunks.sort(key=lambda x: x.chunk.chunk_id)

            # Combine content from multiple chunks of the same document
            combined_content = []
            for result in chunks:
                chunk = result.chunk
                combined_content.append(chunk.content)

            # Join chunks with some indication of boundaries
            document_content = "\n\n---\n\n".join(combined_content)
            context_parts.append(document_content)
            context_parts.append("")  # Add spacing between documents

        context = "\n".join(context_parts)

        # Step 4: Format conversation history
        history_text = ""
        if conversation_history:
            recent = conversation_history[-3:]  # Last 3 exchanges
            history_parts = ["## Previous conversation:"]
            for exchange in recent:
                if exchange.get("user"):
                    history_parts.append(f"**User:** {exchange['user']}")
                if exchange.get("assistant"):
                    history_parts.append(
                        f"**Assistant:** {exchange['assistant'][:200]}..."
                    )
            history_text = "\n".join(history_parts)

        # Step 5: Generate response
        logger.info("📍 Generating enhanced RAG response...")
        return self._get_enhanced_response(question, context, history_text)

    def _get_basic_response(self, question: str):
        """Generate basic response for non-relevant questions"""
        messages = [
            {
                "role": "system",
                "content": """You are a Vietnamese legal assistant. The user's question is not related to prosecution office documents. 

Respond politely in Vietnamese and suggest they ask about:
- Prosecution office addresses
- Contact information  
- Legal authority and jurisdiction
- Legal regulations

🌟 Developed by Hoàng Yến 🌟""",
            },
            {"role": "user", "content": question},
        ]

        return self._stream_openai_response(messages)

    def _get_no_docs_response(self, question: str):
        """Generate response when no relevant documents found"""
        messages = [
            {
                "role": "system",
                "content": """You are a Vietnamese legal assistant. The question is related to legal documents but no specific information was found in the database.

Respond politely in Vietnamese and suggest the user provide more specific information.

🌟 Developed by Hoàng Yến 🌟""",
            },
            {"role": "user", "content": question},
        ]

        return self._stream_openai_response(messages)

    def _get_enhanced_response(self, question: str, context: str, history: str):
        """Generate enhanced response with document context"""
        system_prompt = f"""You are a professional Vietnamese legal assistant developed by Hoàng Yến.

🌟 VIETNAMESE LEGAL AI ASSISTANT DEVELOPED BY HOÀNG YẾN 🌟

PROFESSIONAL STANDARDS:
• Clear, concise, and well-structured responses
• Use appropriate legal terminology when necessary
• Respectful and professional tone
• Reference previous conversation when relevant

RESPONSE FORMAT REQUIREMENTS:
• **Use Markdown formatting** for readability
• **Bold important terms, concepts, and key points**
• Use appropriate emojis (⚖️ for legal issues, 📋 for documents, ⚠️ for warnings, etc.)

{history}

Based on the following legal documents (retrieved using semantic search and re-ranking), answer the user's question:

{context}

Current question: {question}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        return self._stream_openai_response(messages)

    def _stream_openai_response(self, messages):
        """Stream response from OpenAI"""
        try:
            stream = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0,
                stream=True,
                max_tokens=4086,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"⚠️ Xin lỗi, đã có lỗi xảy ra: {str(e)}"


# Initialize enhanced RAG agent lazily
_enhanced_agent = None


def get_enhanced_agent():
    """Lazy initialization of enhanced RAG agent"""
    global _enhanced_agent
    if _enhanced_agent is None:
        logger.info("🎬 Starting Enhanced RAG Agent initialization...")
        _enhanced_agent = EnhancedRAGAgent()
        logger.info("🌟 Enhanced RAG Agent is ready to serve requests!")
    return _enhanced_agent


def get_enhanced_streaming_response(
    question: str, conversation_history: List[Dict[str, str]] = None
):
    """Public function to get enhanced streaming response"""
    agent = get_enhanced_agent()
    return agent.get_streaming_response(question, conversation_history)
