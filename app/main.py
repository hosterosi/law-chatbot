import os
import json
import logging
from datetime import datetime
from flask import Flask, request, render_template, Response, stream_with_context, jsonify, send_file
import sys
import io
import base64
from werkzeug.utils import secure_filename
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory containing main.py
base_dir = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, "../templates"),
    static_folder=os.path.join(base_dir, "../static"),
)

# Global variable to track which RAG function is loaded
_current_rag_function = None
_rag_function_name = None


# Lazy import function to avoid loading heavy dependencies on Vercel
def get_rag_function():
    """Dynamically import the appropriate RAG function based on environment"""
    global _current_rag_function, _rag_function_name

    # Return cached function if available
    if _current_rag_function is not None:
        logger.info(f"🔄 Using cached RAG function: {_rag_function_name}")
        return _current_rag_function

    # Check if we're in Vercel environment
    is_vercel = os.getenv("VERCEL") or os.getenv("VERCEL_ENV")
    logger.info(f"🌍 Environment check - VERCEL: {is_vercel}")

    try:
        # Try enhanced RAG first
        logger.info("🔧 Attempting to import Enhanced RAG...")

        try:
            from .enhanced_rag_agent import get_enhanced_streaming_response

            logger.info("✅ Enhanced RAG imported successfully")
            _current_rag_function = get_enhanced_streaming_response
            _rag_function_name = (
                "Enhanced RAG (Vercel-optimized)" if is_vercel else "Enhanced RAG"
            )
            return get_enhanced_streaming_response
        except ImportError:
            # Fallback for when relative imports don't work
            import sys

            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from enhanced_rag_agent import get_enhanced_streaming_response

            logger.info("✅ Enhanced RAG imported successfully (fallback path)")
            _current_rag_function = get_enhanced_streaming_response
            _rag_function_name = (
                "Enhanced RAG (Vercel-optimized fallback)"
                if is_vercel
                else "Enhanced RAG (fallback)"
            )
            return get_enhanced_streaming_response

    except Exception as e:
        logger.error(f"❌ Enhanced RAG import failed: {e}")

        try:
            # Fallback to simple RAG
            logger.info("🔧 Falling back to Simple RAG...")

            try:
                from .simple_rag_agent import get_simple_streaming_response
            except ImportError:
                import sys

                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                from simple_rag_agent import get_simple_streaming_response

            logger.info("✅ Simple RAG imported successfully")
            _current_rag_function = get_simple_streaming_response
            _rag_function_name = "Simple RAG (fallback)"
            return get_simple_streaming_response

        except Exception as e2:
            logger.error(f"❌ Simple RAG also failed: {e2}")
            logger.info("🔄 Falling back to basic OpenAI response")
            _current_rag_function = get_basic_openai_response()
            _rag_function_name = "Basic OpenAI (fallback)"
            return get_basic_openai_response()


def get_fallback_response():
    """Simple fallback response when RAG system is not available"""

    def fallback_stream(question, conversation_history=None):
        yield "⚠️ Xin lỗi, hệ thống tạm thời không khả dụng. "
        yield "Vui lòng thử lại sau hoặc liên hệ hỗ trợ kỹ thuật.\n\n"
        yield "🌟 Developed by Hoàng Yến 🌟"

    return fallback_stream


def get_basic_openai_response():
    """Fallback to basic OpenAI response when enhanced/simple RAG fails"""

    def basic_stream(question, conversation_history=None):
        try:
            import openai
        except ImportError:
            logger.error("❌ OpenAI package not available for basic response")
            fallback = get_fallback_response()
            for chunk in fallback(question, conversation_history):
                yield chunk
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY not found for basic response")
            fallback = get_fallback_response()
            for chunk in fallback(question, conversation_history):
                yield chunk
            return

        try:
            # Initialize client without proxies parameter to avoid compatibility issues
            client = openai.OpenAI(api_key=api_key)
        except TypeError as e:
            if "proxies" in str(e):
                logger.error(
                    "❌ OpenAI client initialization failed due to proxies parameter"
                )
                # Try fallback initialization
                try:
                    client = openai.OpenAI(api_key=api_key)
                except Exception:
                    fallback = get_fallback_response()
                    for chunk in fallback(question, conversation_history):
                        yield chunk
                    return
            else:
                logger.error(f"❌ OpenAI client initialization failed: {e}")
                fallback = get_fallback_response()
                for chunk in fallback(question, conversation_history):
                    yield chunk
                return

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI pháp lý Việt Nam được phát triển bởi Hoàng Yến. Trả lời ngắn gọn và hữu ích.",
                    },
                    {"role": "user", "content": question},
                ],
                stream=True,
                max_tokens=500,
                temperature=0,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"❌ Basic OpenAI streaming failed: {e}")
            fallback = get_fallback_response()
            for chunk in fallback(question, conversation_history):
                yield chunk

    return basic_stream


@app.route("/")
def index():
    """Main application page"""
    return render_template("index.html")


@app.route("/convert")
def convert_page():
    """Document conversion page"""
    return render_template("convert.html")


@app.route("/explorer")
def explorer_page():
    """File explorer page for raw_data_md"""
    return render_template("explorer.html")


@app.route("/ask", methods=["POST"])
@app.route("/ask-enhanced", methods=["POST"])
def ask():
    """
    Handles the user's question and streams back the response using enhanced RAG with router LLM.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = request.get_json()
    question = data.get("question")
    conversation_history = data.get("conversation_history", [])

    logger.info(f"\n🌐 FLASK REQUEST at {timestamp}")
    logger.info(f"📨 Received question: {question}")
    logger.info(f"📜 Conversation history length: {len(conversation_history)}")
    logger.info(f"🔗 Client IP: {request.remote_addr}")

    if not question:
        logger.error("❌ No question provided in request")
        return Response("Error: Question is required.", status=400)

    def generate():
        """
        A generator function that yields Server-Sent Events for enhanced RAG.
        """
        try:
            logger.info("🔄 Getting RAG function...")
            rag_function = get_rag_function()

            # Log which RAG system is being used
            logger.info(f"🤖 Using RAG system: {_rag_function_name}")

            if rag_function is None:
                logger.error("❌ RAG function is not available.")
                fallback_stream = get_fallback_response()
                for chunk in fallback_stream(question, conversation_history):
                    sse_data = json.dumps({"token": chunk})
                    yield f"data: {sse_data}\n\n"
                return

            logger.info("🔄 Starting RAG streaming...")

            # Check if rag_function is callable or needs to be called
            if callable(rag_function):
                stream = rag_function(question, conversation_history)
            else:
                # If it's already a generator, use it directly
                stream = rag_function

            for chunk in stream:
                sse_data = json.dumps({"token": chunk})
                yield f"data: {sse_data}\n\n"

        except Exception as e:
            logger.error(f"❌ RAG streaming error: {str(e)}")
            # Send error message if RAG fails
            error_data = json.dumps({"error": f"RAG error: {str(e)}"})
            yield f"data: {error_data}\n\n"

    # Return a streaming response
    logger.info("📡 Returning SSE response stream")
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# Force reload RAG function if needed
@app.route("/reload-rag", methods=["POST"])
def reload_rag():
    """Force reload the RAG function - useful for development"""
    global _current_rag_function, _rag_function_name

    logger.info("🔄 Force reloading RAG function...")
    _current_rag_function = None
    _rag_function_name = None

    # Test the reload
    rag_function = get_rag_function()

    return {
        "status": "success",
        "message": f"RAG function reloaded successfully",
        "current_rag": _rag_function_name,
    }


@app.route("/status", methods=["GET"])
def status():
    """Get current system status and RAG configuration"""
    global _current_rag_function, _rag_function_name

    # Get RAG function info (without initializing if not already done)
    rag_info = _rag_function_name if _current_rag_function else "Not loaded"

    # Check environment
    is_vercel = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))

    # Check file availability
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    data_path = os.path.join(project_root, "data")
    rules_path = os.path.join(project_root, "raw_data", "rules.txt")
    cache_path = os.path.join(current_dir, ".cache", "embeddings_cache.pkl")

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_system": {"active": rag_info, "cached": _current_rag_function is not None},
        "environment": {
            "vercel": is_vercel,
            "has_openai_key": has_openai_key,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "files": {
            "data_directory_exists": os.path.exists(data_path),
            "rules_file_exists": os.path.exists(rules_path),
            "embeddings_cache_exists": os.path.exists(cache_path),
            "num_md_files": len([f for f in os.listdir(data_path) if f.endswith(".md")])
            if os.path.exists(data_path)
            else 0,
        },
        "features": {
            "enhanced_rag": "available"
            if rag_info and "Enhanced" in rag_info
            else "not_active",
            "simple_rag": "fallback_available",
            "embeddings": "openai_text_embedding_3_small",
            "chunking": "overlap_based",
            "reranking": "multi_factor",
        },
    }


# Document conversion endpoints
@app.route("/api/convert", methods=["POST"])
def convert_document():
    """Convert uploaded document to markdown"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Check file extension
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

        supported_extensions = ['pdf', 'txt', 'docx', 'doc']
        if file_extension not in supported_extensions:
            return jsonify({"error": f"Unsupported file type. Supported: {', '.join(supported_extensions)}"}), 400

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Import document processing function
            from .document_processor import process_document

            # Process document
            result = process_document(temp_path, filename)

            return jsonify(result)

        except ImportError:
            return jsonify({"error": "Document processing module not available"}), 500
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Document conversion error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/save-markdown", methods=["POST"])
def save_markdown():
    """Save converted markdown to raw_data_md directory"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        content = data.get('content')

        if not filename or not content:
            return jsonify({"error": "Filename and content are required"}), 400

        # Ensure raw_data_md directory exists
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_md_dir = os.path.join(base_dir, 'raw_data_md')
        os.makedirs(raw_data_md_dir, exist_ok=True)

        # Clean filename and add .md extension if not present
        clean_filename = secure_filename(filename)
        if not clean_filename.endswith('.md'):
            clean_filename += '.md'

        filepath = os.path.join(raw_data_md_dir, clean_filename)

        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return jsonify({
            "success": True,
            "message": f"File saved successfully: {clean_filename}",
            "filepath": filepath
        })

    except Exception as e:
        logger.error(f"Save markdown error: {e}")
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500


@app.route("/api/files", methods=["GET"])
def list_files():
    """List files in raw_data_md directory"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_md_dir = os.path.join(base_dir, 'raw_data_md')

        if not os.path.exists(raw_data_md_dir):
            return jsonify({"files": []})

        files = []
        for filename in os.listdir(raw_data_md_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(raw_data_md_dir, filename)
                stat = os.stat(filepath)
                files.append({
                    "name": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": filepath
                })

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({"files": files})

    except Exception as e:
        logger.error(f"List files error: {e}")
        return jsonify({"error": "Failed to list files"}), 500


@app.route("/api/files/<filename>", methods=["GET"])
def get_file_content(filename):
    """Get content of a specific markdown file"""
    try:
        clean_filename = secure_filename(filename)
        if not clean_filename.endswith('.md'):
            return jsonify({"error": "Only markdown files are supported"}), 400

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_md_dir = os.path.join(base_dir, 'raw_data_md')
        filepath = os.path.join(raw_data_md_dir, clean_filename)

        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        stat = os.stat(filepath)
        return jsonify({
            "filename": clean_filename,
            "content": content,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })

    except Exception as e:
        logger.error(f"Get file content error: {e}")
        return jsonify({"error": "Failed to read file"}), 500


@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    """Delete a specific markdown file"""
    try:
        clean_filename = secure_filename(filename)
        if not clean_filename.endswith('.md'):
            return jsonify({"error": "Only markdown files can be deleted"}), 400

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_md_dir = os.path.join(base_dir, 'raw_data_md')
        filepath = os.path.join(raw_data_md_dir, clean_filename)

        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        os.unlink(filepath)

        return jsonify({
            "success": True,
            "message": f"File deleted successfully: {clean_filename}"
        })

    except Exception as e:
        logger.error(f"Delete file error: {e}")
        return jsonify({"error": "Failed to delete file"}), 500


if __name__ == "__main__":
    logger.info("🚀 Starting Flask application...")
    logger.info("🔄 Pre-loading RAG function...")

    # Pre-load RAG function to show which one is being used
    rag_func = get_rag_function()
    logger.info(f"✅ Pre-loaded RAG system: {_rag_function_name}")

    app.run(debug=True, host="0.0.0.0", port=5001)


# Export for Vercel - Flask WSGI app
# Vercel's Python runtime expects either:
# 1. A 'handler' class that inherits from BaseHTTPRequestHandler, or
# 2. An 'app' variable that exposes a WSGI/ASGI application
# We're using option 2 with Flask (WSGI)

# Export both 'app' and 'application' for compatibility
application = app
# The 'app' variable is already defined above, Vercel will detect it
