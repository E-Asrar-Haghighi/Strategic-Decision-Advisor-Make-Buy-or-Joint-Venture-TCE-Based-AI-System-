import os
from dotenv import load_dotenv
import logging
from agents.context_extractor import ContextExtractorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables!")
        return
    
    # Mask the API key for logging
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    logger.info(f"API Key loaded: {masked_key}")
    
    # Test LLM connectivity
    try:
        context_agent = ContextExtractorAgent()
        if context_agent.test_llm():
            logger.info("✅ LLM test successful! The API key is working correctly.")
        else:
            logger.error("❌ LLM test failed. Please check your API key and network connection.")
    except Exception as e:
        logger.error(f"❌ Error during LLM test: {e}")

if __name__ == "__main__":
    main() 