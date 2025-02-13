import os
import logging
from typing import Dict, Optional, List
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
from langchain.schema import HumanMessage, SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatAgent:
    def __init__(self):
        """Initialize the chat agent with Llama 70B model."""
        sambanova_api_key = "540f8914-997e-46c6-829a-ff76f5d4d265"
        os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key
        
        self.llm = ChatSambaNovaCloud(
            model="llama3-70b",
            temperature=0.7,
            max_tokens=4000,
            sambanova_api_key=sambanova_api_key
        )
        
    async def process_message(self, message: str, context: Dict = None) -> Dict[str, any]:
        """
        Process a message in the chat conversation.
        
        Args:
            message: The user's input message
            context: Optional context for the conversation
            
        Returns:
            Dict containing the response and status
        """
        try:
            # Log the incoming message
            logger.info(f"Processing message: {message[:100]}...")
            
            # Create messages
            messages = []
            if context:
                messages.append(SystemMessage(content=f"Context: {context}"))
            messages.append(HumanMessage(content=message))
            
            # Generate response using Llama
            response = await self.llm.agenerate(messages=[messages])
            response_text = response.generations[0][0].text.strip()
            
            return {
                "response": response_text,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error processing your message.",
                "success": False,
                "error": str(e)
            }
            
    async def generate_suggestions(self, message: str) -> List[str]:
        """
        Generate follow-up suggestions based on the current message.
        
        Args:
            message: The current message
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            prompt = f"""Based on this conversation: "{message}"
            Generate 3 relevant follow-up questions that would help continue the conversation.
            Keep the questions concise and natural. Format them as a simple list."""
            
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.agenerate(messages=[messages])
            suggestions_text = response.generations[0][0].text.strip()
            
            # Parse suggestions - assuming they come as a list with newlines
            suggestions = [q.strip().lstrip('- ').lstrip('1234567890.)')
                         for q in suggestions_text.split('\n')
                         if q.strip() and not q.strip().startswith('#')]
            
            # Return at most 3 suggestions
            return suggestions[:3] if suggestions else [
                "Tell me more about that",
                "What are your thoughts on this?",
                "Could you elaborate?"
            ]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return [
                "Tell me more about that",
                "What are your thoughts on this?",
                "Could you elaborate?"
            ]

if __name__ == "__main__":
    import asyncio
    res = asyncio.run(ChatAgent().process_message("What is the capital of France?"))
    print(res)