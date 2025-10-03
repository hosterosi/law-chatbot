import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

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


class RelevanceCheck(BaseModel):
    """Model for relevance checking response"""

    relevant: bool = Field(
        description="Whether the query is relevant to the legal documents"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the relevance decision")


class SimpleRAGAgent:
    def __init__(self):
        logger.info("🚀 Initializing Simple RAG Agent for Vercel...")

        # Check if OpenAI is available
        if openai is None:
            logger.error("❌ OpenAI package not available")
            raise ImportError("OpenAI package is required but not installed")

        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY is required")

        # Initialize OpenAI client with error handling for compatibility issues
        self.client = None
        try:
            # Try initialization without any extra parameters first
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("✅ OpenAI client initialized successfully")
        except TypeError as e:
            if "proxies" in str(e):
                logger.error(
                    "❌ OpenAI client failed due to proxies parameter - trying alternative initialization"
                )
                try:
                    # Alternative initialization approach
                    self.client = openai.OpenAI(api_key=api_key)
                    logger.info("✅ OpenAI client initialized with alternative method")
                except Exception as alt_e:
                    logger.error(
                        f"❌ Alternative OpenAI initialization also failed: {alt_e}"
                    )
                    raise RuntimeError(f"Cannot initialize OpenAI client: {alt_e}")
            else:
                logger.error(
                    f"❌ OpenAI client initialization failed with TypeError: {e}"
                )
                raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise

        if self.client is None:
            raise RuntimeError("Failed to initialize OpenAI client")

        # Load rules.txt and documents
        self.rules_content = self._load_rules()
        self.documents = self._load_documents()

        logger.info("🎉 Simple RAG Agent initialization complete!")

    def _load_rules(self) -> str:
        """Load rules.txt content"""
        # More robust path resolution that works from any starting point
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up from app/ to project root
        rules_path = os.path.join(project_root, "raw_data", "rules.txt")

        logger.info(f"📁 Loading rules from: {rules_path}")
        logger.info(f"🗂️ Current file location: {current_dir}")
        logger.info(f"📂 Project root: {project_root}")

        try:
            if not os.path.exists(rules_path):
                logger.warning(f"⚠️ Rules file not found at: {rules_path}")
                # Try alternative path in case we're running from different location
                alt_rules_path = os.path.join(os.getcwd(), "raw_data", "rules.txt")
                logger.info(f"🔄 Trying alternative path: {alt_rules_path}")

                if os.path.exists(alt_rules_path):
                    rules_path = alt_rules_path
                    logger.info("✅ Found rules file at alternative path")
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

    def _load_documents(self) -> Dict[str, str]:
        """Load all documents into memory"""
        documents = {}
        # More robust path resolution that works from any starting point
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up from app/ to project root
        data_path = os.path.join(project_root, "data")

        logger.info(f"📂 Loading documents from: {data_path}")
        logger.info(f"🗂️ Current working directory: {os.getcwd()}")

        try:
            if not os.path.exists(data_path):
                logger.warning(f"⚠️ Data directory not found at: {data_path}")
                # Try alternative path in case we're running from different location
                alt_data_path = os.path.join(os.getcwd(), "data")
                logger.info(f"🔄 Trying alternative path: {alt_data_path}")

                if os.path.exists(alt_data_path):
                    data_path = alt_data_path
                    logger.info("✅ Found data directory at alternative path")
                else:
                    return {}

            md_files = [f for f in os.listdir(data_path) if f.endswith(".md")]
            logger.info(f"📄 Found {len(md_files)} .md files")

            for filename in md_files:
                file_path = os.path.join(data_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents[filename] = content
                        logger.info(f"📖 Loaded {filename}, size: {len(content)} chars")
                except Exception as e:
                    logger.error(f"❌ Error reading {filename}: {e}")

            logger.info(f"✅ Loaded {len(documents)} documents successfully")
            return documents

        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
            return {}

    def _check_relevance(self, question: str) -> RelevanceCheck:
        """Check if question is relevant using OpenAI"""
        logger.info(f"🤔 Checking relevance for question: {question[:100]}...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a legal assistant. Determine if the user's question is relevant to Vietnamese prosecution office documents.

Available documents:
{self.rules_content}

Respond ONLY with valid JSON in this format:
{{"relevant": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}""",
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()
            logger.info(f"🔍 Relevance response: {result_text}")

            # Clean up markdown code blocks if present
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                result_text = '\n'.join(json_lines).strip()

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

    def _find_relevant_documents(
        self, question: str, conversation_history: List[Dict[str, str]]
    ) -> List[str]:
        """Find relevant documents using OpenAI"""
        logger.info("🔍 Finding relevant documents...")

        # Create context from conversation history
        context_parts = []
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 4 exchanges
            for exchange in recent_history:
                if exchange.get("user"):
                    context_parts.append(f"User: {exchange['user']}")
                if exchange.get("assistant"):
                    context_parts.append(f"Assistant: {exchange['assistant'][:200]}...")

        context_text = (
            "\n".join(context_parts) if context_parts else "No previous conversation."
        )

        # Get document list
        doc_list = "\n".join(
            [
                f"- {filename}: {self.rules_content.split(filename + ':')[1].split('\n')[0] if filename + ':' in self.rules_content else 'Legal document'}"
                for filename in self.documents.keys()
            ]
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a document retrieval system. Based on the user's question and conversation context, select the most relevant documents.

Available documents:
{doc_list}

Return ONLY a JSON array of filenames, like: ["file1.md", "file2.md"]
Select maximum 3 most relevant documents.""",
                    },
                    {
                        "role": "user",
                        "content": f"Conversation context:\n{context_text}\n\nCurrent question: {question}",
                    },
                ],
                temperature=0,
                max_tokens=200,
            )

            result_text = response.choices[0].message.content.strip()
            logger.info(f"📄 Document selection response: {result_text}")

            # Clean up markdown code blocks if present
            if result_text.startswith('```'):
                # Extract JSON from markdown code block
                lines = result_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                result_text = '\n'.join(json_lines).strip()
            
            # Try to parse JSON
            try:
                selected_files = json.loads(result_text)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Failed to parse JSON, trying to extract array: {result_text}")
                # Try to extract array pattern [...]
                import re
                array_match = re.search(r'\[.*?\]', result_text, re.DOTALL)
                if array_match:
                    selected_files = json.loads(array_match.group())
                else:
                    selected_files = []

            # Validate and filter existing documents
            valid_files = [f for f in selected_files if f in self.documents]
            logger.info(
                f"✅ Selected {len(valid_files)} relevant documents: {valid_files}"
            )

            return valid_files

        except Exception as e:
            logger.error(f"❌ Error in document selection: {e}")
            # Fallback: return first 2 documents
            return list(self.documents.keys())[:2]

    def get_streaming_response(
        self, question: str, conversation_history: List[Dict[str, str]] = None
    ):
        """Get streaming response using OpenAI"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"🚀 NEW REQUEST at {timestamp}: {question}")

        if conversation_history is None:
            conversation_history = []

        # Step 1: Check relevance
        relevance_check = self._check_relevance(question)

        if not relevance_check.relevant:
            logger.info("🚫 Question not relevant - providing general response")
            return self._get_basic_response(question)

        # Step 2: Find relevant documents
        relevant_files = self._find_relevant_documents(question, conversation_history)

        if not relevant_files:
            logger.warning("⚠️ No relevant documents found")
            return self._get_no_docs_response(question)

        # Step 3: Prepare context
        context_parts = []
        for filename in relevant_files:
            content = self.documents[filename]
            # Limit content size
            truncated_content = (
                content[:2000] + "..." if len(content) > 2000 else content
            )
            context_parts.append(f"**Document: {filename}**\n{truncated_content}")

        context = "\n\n".join(context_parts)

        # Step 4: Format conversation history
        history_text = ""
        if conversation_history:
            recent = conversation_history[-4:]
            history_parts = ["## Previous conversation:"]
            for exchange in recent:
                if exchange.get("user"):
                    history_parts.append(f"**User:** {exchange['user']}")
                if exchange.get("assistant"):
                    history_parts.append(
                        f"**Assistant:** {exchange['assistant'][:300]}..."
                    )
            history_text = "\n".join(history_parts)

        # Step 5: Generate response
        logger.info("📍 Generating final response...")
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

Based on the following legal documents, answer the user's question:

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
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                stream=True,
                max_tokens=1000,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield f"⚠️ Xin lỗi, đã có lỗi xảy ra: {str(e)}"


# Initialize simple RAG agent lazily
_simple_agent = None


def get_simple_agent():
    """Lazy initialization of simple RAG agent"""
    global _simple_agent
    if _simple_agent is None:
        logger.info("🎬 Starting Simple RAG Agent initialization...")
        _simple_agent = SimpleRAGAgent()
        logger.info("🌟 Simple RAG Agent is ready to serve requests!")
    return _simple_agent


def get_simple_streaming_response(
    question: str, conversation_history: List[Dict[str, str]] = None
):
    """Public function to get simple streaming response"""
    agent = get_simple_agent()
    return agent.get_streaming_response(question, conversation_history)
