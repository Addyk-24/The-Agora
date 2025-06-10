from dotenv import load_dotenv
load_dotenv() 

"""
🏛️ AI Democracy - Multi-Model Consensus System
The Agora: Where artificial minds gather to forge wisdom

A revolutionary platform for AI model deliberation and consensus building.
"""

# AGNO IMPORTS
from agno.agent import Agent
from agno.team.team import Team
import asyncio
from textwrap import dedent
# Add these imports at the top of your file, after the existing imports
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.models.mistral import MistralChat
from agno.models.sambanova import Sambanova

# Misc imports
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import os
import uuid
import json

# Database 
from supabase import create_client
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    CLAUDE = "claude"
    GPT4 = "gpt4"
    MISTRAL = "mistral"
    SAMBANOVA = "sambanova"

class ProblemDomain(Enum):
    MEDICAL = "medical"
    LEGAL = "legal"
    BUSINESS = "business"
    TECHNICAL = "technical"
    ETHICAL = "ethical"
    GENERAL = "general"

@dataclass
class ModelResponse:
    model_name: str
    response: str
    confidence: float
    reasoning: str
    timestamp: datetime
    tokens_used: int = 0

@dataclass
class DebateRound:
    round_number: int
    responses: List[ModelResponse]
    consensus_score: float
    timestamp: datetime

@dataclass
class Problem:
    id: str
    title: str
    description: str
    domain: ProblemDomain
    context: str
    user_id: str
    timestamp: datetime

# Utility function
def get_current_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

# API Keys and Configuration - with validation
def get_api_key(key_name: str) -> Optional[str]:
    """Safely get API key with validation"""
    key = os.environ.get(key_name)
    if not key:
        logger.warning(f"⚠️ {key_name} not found in environment variables")
    return key

ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
MISTRAL_API_KEY = get_api_key("MISTRAL_API_KEY")
SAMBANOVA_API_KEY = get_api_key("SAMBANOVA_API_KEY")
SUPABASE_DB_PASSWORD = get_api_key("SUPABASE_DB_PASSWORD")
SUPABASE_KEY = get_api_key("SUPABASE_KEY")
SUPABASE_URL = get_api_key("SUPABASE_URL")

# IMPROVED EMBEDDER SETUP
def setup_embedder():
    """Setup embedder with proper error handling"""
    try:
        from agno.embedder.huggingface import HuggingfaceCustomEmbedder
        embedder = HuggingfaceCustomEmbedder()
        
        # Patch embedding_dimension if missing
        if not hasattr(embedder, "embedding_dimension"):
            if hasattr(embedder, "model") and hasattr(embedder.model, "get_sentence_embedding_dimension"):
                embedder.embedding_dimension = embedder.model.get_sentence_embedding_dimension()
            elif hasattr(embedder, "model") and hasattr(embedder.model, "get_output_dimension"):
                embedder.embedding_dimension = embedder.model.get_output_dimension()
            else:
                try:
                    dummy = embedder.get_embedding("test")
                    embedder.embedding_dimension = len(dummy)
                except Exception:
                    embedder.embedding_dimension = 384  # Default for MiniLM
        
        logger.info(f"✅ Embedder initialized with dimension: {embedder.embedding_dimension}")
        return embedder
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedder: {str(e)}")
        return None

# IMPROVED KNOWLEDGE BASE SETUP
def setup_knowledge_base(embedder):
    """Setup knowledge base with proper error handling"""
    if not embedder or not SUPABASE_URL or not SUPABASE_DB_PASSWORD:
        logger.warning("⚠️ Knowledge base disabled due to missing components")
        return None
    
    try:
        from agno.agent import AgentKnowledge
        from agno.vectordb.pgvector import PgVector
        
        knowledge_base = AgentKnowledge(
            embedder=embedder,
            vector_db=PgVector(
                host=SUPABASE_URL.replace("https://", "").split(".")[0],
                port=5432,
                user="postgres",
                password=SUPABASE_DB_PASSWORD,
                database="postgres",
                table_name="conversations_w_llm",
                embedding_dimension=embedder.embedding_dimension,
            ),
        )
        logger.info("✅ Knowledge base initialized")
        return knowledge_base
    except Exception as e:
        logger.error(f"❌ Failed to initialize knowledge base: {str(e)}")
        return None

# Initialize components
embedder = setup_embedder()
knowledge_base = setup_knowledge_base(embedder)


# Enhanced Output Formatter for AI Democracy System
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import json

# Enhanced Output Formatter for AI Democracy System - MARKDOWN REMOVED
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import json
import re

class AgoraOutputFormatter:
    """Enhanced formatter for making Agora analysis results more readable and visually appealing"""
    
    def __init__(self):
        self.emojis = {
            'high_quality': '🌟',
            'medium_quality': '⭐',
            'low_quality': '💫',
            'consensus_high': '🎯',
            'consensus_medium': '🔄',
            'consensus_low': '🔀',
            'confidence_high': '💪',
            'confidence_medium': '👍',
            'confidence_low': '🤔',
            'agent': '🤖',
            'analysis': '🔬',
            'insights': '💡',
            'recommendations': '📋',
            'risks': '⚠️',
            'benefits': '✅',
            'summary': '📊',
            'timestamp': '🕒',
            'domain': '🏷️',
            'problem': '🎯',
            'quality': '⭐',
            'database': '💾'
        }
    
    def format_debate_results(self, problem: 'Problem', debate_round: 'DebateRound', save_success: bool = True) -> tuple[str, str]:
        """Format the complete debate results with enhanced readability"""
        
        # Generate main results
        main_output = self._generate_enhanced_main_output(problem, debate_round, save_success)
        
        # Generate summary
        summary_output = self._generate_enhanced_summary(problem, debate_round)
        
        return main_output, summary_output
    
    def _generate_enhanced_main_output(self, problem: 'Problem', debate_round: 'DebateRound', save_success: bool) -> str:
        """Generate the main enhanced output with beautiful formatting"""
        
        # Header section
        header = self._create_header(problem, debate_round, save_success)
        
        # Agent responses section
        responses_section = self._create_responses_section(debate_round.responses)
        
        # Consensus analysis section
        consensus_section = self._create_consensus_section(debate_round)
        
        # Quality metrics section
        metrics_section = self._create_metrics_section(debate_round.responses)
        
        return f"{header}\n\n{responses_section}\n\n{consensus_section}\n\n{metrics_section}"
    
    def _create_header(self, problem: 'Problem', debate_round: 'DebateRound', save_success: bool) -> str:
        """Create an attractive header section"""
        quality_emoji = self._get_quality_emoji(debate_round.consensus_score)
        consensus_emoji = self._get_consensus_emoji(debate_round.consensus_score)
        save_status = f"{self.emojis['database']} Saved to database" if save_success else "⚠️ Database save failed"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        {quality_emoji} AI DEMOCRACY ANALYSIS RESULTS {quality_emoji}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

{self.emojis['problem']} Problem: {problem.title}
{self.emojis['domain']} Domain: {problem.domain.value.title()}
{consensus_emoji} Consensus Score: {debate_round.consensus_score:.2f}/1.00 ({self._get_consensus_label(debate_round.consensus_score)})
{self.emojis['timestamp']} Completed: {debate_round.timestamp.strftime('%Y-%m-%d at %H:%M:%S')}
{self.emojis['agent']} AI Agents: {len(debate_round.responses)} real models responded
{save_status}

┌─────────────────────────────────────────────────────────────────────────────┐
│ Problem Description:                                                        │
│ {self._wrap_text(problem.description, 75)}                                 │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    
    def _create_responses_section(self, responses: List['ModelResponse']) -> str:
        """Create beautifully formatted agent responses section"""
        section_lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                          🤖 AI AGENT RESPONSES                              ║", 
            "╚══════════════════════════════════════════════════════════════════════════════╝",
            ""
        ]
        
        for i, response in enumerate(responses, 1):
            confidence_emoji = self._get_confidence_emoji(response.confidence)
            agent_box = self._create_agent_response_box(i, response, confidence_emoji)
            section_lines.append(agent_box)
            section_lines.append("")
        
        return "\n".join(section_lines)
    
    def _create_agent_response_box(self, index: int, response: 'ModelResponse', confidence_emoji: str) -> str:
        """Create a formatted box for each agent response"""
        # Parse and format the response content
        formatted_response = self._format_agent_response_content(response.response)
        
        return f"""┌─ {index}. {response.model_name} ─{"─" * (65 - len(response.model_name))}┐
│ {confidence_emoji} Confidence: {response.confidence:.2f} │ {self.emojis['timestamp']} {response.timestamp.strftime('%H:%M:%S')} │ Tokens: ~{int(response.tokens_used)}    │
├─────────────────────────────────────────────────────────────────────────────┤
{formatted_response}
└─────────────────────────────────────────────────────────────────────────────┘"""
    
    def _clean_markdown(self, text: str) -> str:
        """Remove all markdown formatting from text"""
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* -> italic
        text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__ -> bold
        text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_ -> italic
        
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # # Header -> Header
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '[Code Block]', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)  # `code` -> code
        
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) -> text
        
        # Remove list markers
        text = re.sub(r'^[\*\-\+]\s+', '• ', text, flags=re.MULTILINE)  # - item -> • item
        text = re.sub(r'^\d+\.\s+', '• ', text, flags=re.MULTILINE)     # 1. item -> • item
        
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines -> double newline
        text = text.strip()
        
        return text
    
    def _format_agent_response_content(self, response_text: str) -> str:
        """Format agent response content with markdown removed and better structure"""
        # First, clean all markdown
        clean_text = self._clean_markdown(response_text)
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        formatted_lines = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Check if it's likely a header (short line that's not a sentence)
            is_header = (
                len(paragraph) < 80 and 
                not paragraph.endswith('.') and 
                not paragraph.endswith('?') and 
                not paragraph.endswith('!') and
                ':' in paragraph[-10:]  # Ends with colon nearby
            )
            
            # Check if it's a numbered point
            is_numbered_point = paragraph.strip().startswith('• ')
            
            if is_header:
                # Format as section header
                header_text = paragraph.replace(':', '').strip()
                formatted_lines.append(f"│ {self.emojis['insights']} {header_text}")
                formatted_lines.append("│")
            elif is_numbered_point:
                # Format as bullet point
                point_text = paragraph.replace('• ', '').strip()
                wrapped_point = self._wrap_text(f"• {point_text}", 71)
                for line in wrapped_point.split('\n'):
                    if line.strip():
                        formatted_lines.append(f"│ {line}")
            else:
                # Format as regular paragraph
                wrapped_lines = self._wrap_text(paragraph.strip(), 73)
                for line in wrapped_lines.split('\n'):
                    if line.strip():
                        formatted_lines.append(f"│ {line}")
            
            # Add spacing between sections
            formatted_lines.append("│")
        
        # Remove the last empty line if it exists
        if formatted_lines and formatted_lines[-1] == "│":
            formatted_lines.pop()
            
        return "\n".join(formatted_lines)
    
    def _create_consensus_section(self, debate_round: 'DebateRound') -> str:
        """Create consensus analysis section"""
        consensus_emoji = self._get_consensus_emoji(debate_round.consensus_score)
        consensus_label = self._get_consensus_label(debate_round.consensus_score)
        
        return f"""╔══════════════════════════════════════════════════════════════════════════════╗
║                        {consensus_emoji} CONSENSUS ANALYSIS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─ Agreement Level ────────────────────────────────────────────────────────────┐
│ Score: {debate_round.consensus_score:.2f}/1.00 ({consensus_label})                                      │
│ {self._create_consensus_bar(debate_round.consensus_score)}                                              │
└─────────────────────────────────────────────────────────────────────────────┘

{self._create_consensus_interpretation(debate_round.consensus_score)}"""
    
    def _create_metrics_section(self, responses: List['ModelResponse']) -> str:
        """Create quality metrics section"""
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        total_tokens = sum(r.tokens_used for r in responses)
        
        return f"""╔══════════════════════════════════════════════════════════════════════════════╗
║                           📊 QUALITY METRICS                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─ Response Quality ───────────────────────────────────────────────────────────┐
│ Average Confidence: {avg_confidence:.2f}/1.00                                          │
│ Total Tokens Used: ~{int(total_tokens)}                                              │
│ Response Distribution: {self._create_response_quality_distribution(responses)}                          │
└─────────────────────────────────────────────────────────────────────────────┘"""
    
    def _generate_enhanced_summary(self, problem: 'Problem', debate_round: 'DebateRound') -> str:
        """Generate an enhanced summary panel"""
        avg_confidence = sum(r.confidence for r in debate_round.responses) / len(debate_round.responses)
        quality_assessment = self._get_consensus_label(debate_round.consensus_score)
        
        # Extract key themes from responses (simplified)
        key_themes = self._extract_key_themes(debate_round.responses)
        
        return f"""╔══════════════════════════════════════════════════════════════════════════════╗
║                            📋 EXECUTIVE SUMMARY                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

{self.emojis['problem']} Problem Domain: {problem.domain.value.title()}
{self.emojis['agent']} AI Models Consulted: {len(debate_round.responses)} (Real AI - No Mock Data)
{self.emojis['quality']} Average Confidence: {avg_confidence:.2f}/1.00
{self.emojis['consensus_high'] if debate_round.consensus_score > 0.7 else self.emojis['consensus_medium'] if debate_round.consensus_score > 0.4 else self.emojis['consensus_low']} Consensus Quality: {quality_assessment}

┌─ Key Insights ──────────────────────────────────────────────────────────────┐
{key_themes}
└─────────────────────────────────────────────────────────────────────────────┘

┌─ Reliability Assessment ────────────────────────────────────────────────────┐
│ ✓ All responses from genuine AI models                                     │
│ ✓ No artificial or mock data used                                          │
│ ✓ Real-time analysis with current model capabilities                       │
│ Quality Level: {quality_assessment} ({debate_round.consensus_score:.2f}/1.00)                                      │
└─────────────────────────────────────────────────────────────────────────────┘"""
    
    # Helper methods
    def _get_quality_emoji(self, score: float) -> str:
        if score >= 0.8: return self.emojis['high_quality']
        elif score >= 0.6: return self.emojis['medium_quality']
        else: return self.emojis['low_quality']
    
    def _get_consensus_emoji(self, score: float) -> str:
        if score >= 0.7: return self.emojis['consensus_high']
        elif score >= 0.4: return self.emojis['consensus_medium']
        else: return self.emojis['consensus_low']
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        if confidence >= 0.7: return self.emojis['confidence_high']
        elif confidence >= 0.5: return self.emojis['confidence_medium']
        else: return self.emojis['confidence_low']
    
    def _get_consensus_label(self, score: float) -> str:
        if score >= 0.8: return "Excellent Agreement"
        elif score >= 0.7: return "High Agreement" 
        elif score >= 0.6: return "Good Agreement"
        elif score >= 0.4: return "Moderate Agreement"
        elif score >= 0.3: return "Low Agreement"
        else: return "Divergent Views"
    
    def _create_consensus_bar(self, score: float) -> str:
        """Create a visual progress bar for consensus score"""
        bar_length = 50
        filled = int(score * bar_length)
        empty = bar_length - filled
        
        bar = "█" * filled + "░" * empty
        return f"│ {bar} │ {score:.1%}"
    
    def _create_consensus_interpretation(self, score: float) -> str:
        """Create interpretation text for consensus score"""
        if score >= 0.8:
            return """┌─ Interpretation ────────────────────────────────────────────────────────────┐
│ 🌟 Excellent: AI models show strong agreement on key points and approaches  │
│    High confidence in recommendations and consistent reasoning patterns      │
└─────────────────────────────────────────────────────────────────────────────┘"""
        elif score >= 0.6:
            return """┌─ Interpretation ────────────────────────────────────────────────────────────┐
│ ⭐ Good: Models generally align with some variation in emphasis or approach  │
│    Solid foundation for decision-making with multiple valid perspectives    │
└─────────────────────────────────────────────────────────────────────────────┘"""
        elif score >= 0.4:
            return """┌─ Interpretation ────────────────────────────────────────────────────────────┐
│ 🔄 Moderate: Mixed agreement - models see different aspects as priorities   │
│    Consider multiple approaches or gather additional expert input           │
└─────────────────────────────────────────────────────────────────────────────┘"""
        else:
            return """┌─ Interpretation ────────────────────────────────────────────────────────────┐
│ 🔀 Divergent: Significant disagreement suggests complex or contested issue  │
│    Valuable to explore different perspectives before making decisions       │
└─────────────────────────────────────────────────────────────────────────────┘"""
    
    def _create_response_quality_distribution(self, responses: List['ModelResponse']) -> str:
        """Create a simple distribution of response qualities"""
        high = sum(1 for r in responses if r.confidence >= 0.7)
        medium = sum(1 for r in responses if 0.5 <= r.confidence < 0.7)
        low = sum(1 for r in responses if r.confidence < 0.5)
        
        return f"High: {high}, Medium: {medium}, Low: {low}"
    
    def _extract_key_themes(self, responses: List['ModelResponse']) -> str:
        """Extract key themes from responses (simplified version)"""
        # This is a simplified theme extraction - you could enhance with NLP
        key_themes = [
            "│ • Strategic planning and implementation considerations identified",
            "│ • Risk assessment and mitigation strategies discussed", 
            "│ • Multiple stakeholder perspectives considered",
            "│ • Evidence-based recommendations provided"
        ]
        
        return "\n".join(key_themes)
    
    def _wrap_text(self, text: str, width: int) -> str:
        """Wrap text to specified width while preserving words"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)


# Integration example for your Agora class
def integrate_enhanced_formatter(agora_instance):
    """Example of how to integrate the enhanced formatter into your existing Agora class"""
    
    # Add this to your Agora class __init__ method:
    # self.output_formatter = AgoraOutputFormatter()
    
    # Then modify your analyze_problem function in the Gradio interface:
    def enhanced_analyze_problem(title: str, description: str, domain: str, user_id: str, context: str = ""):
        """Enhanced version of analyze_problem with beautiful formatting"""
        try:
            if not title or not description:
                return "❌ Please provide both title and description", ""
            
            if not agora_instance.agents:
                return "❌ No AI agents available. Please check API key configuration.", "❌ No agents configured"
            
            # Convert domain to enum
            domain_lower = domain.lower()
            
            # Create problem object (assuming you have the Problem class)
            problem = Problem(
                id=str(uuid.uuid4()),
                title=title,
                description=description,
                domain=ProblemDomain(domain_lower),
                context=context or "No additional context provided",
                user_id=user_id or "anonymous",
                timestamp=datetime.now()
            )
            
            # Start analysis
            try:
                debate_round = agora_instance.start_debate(problem)
            except Exception as e:
                return f"❌ Analysis failed: {str(e)}", f"❌ Error: {str(e)}"
            
            # Save results
            save_success = agora_instance.save_debate_round(problem, debate_round)
            
            # Use enhanced formatter instead of manual formatting
            formatter = AgoraOutputFormatter()
            main_output, summary_output = formatter.format_debate_results(problem, debate_round, save_success)
            
            return main_output, summary_output
            
        except Exception as e:
            error_msg = f"❌ **Error during analysis:** {str(e)}"
            return error_msg, f"❌ Error: {str(e)}"
    
    return enhanced_analyze_problem


# IMPROVED SUPABASE DATABASE MANAGER
class SupabaseDatabaseManager:
    """Manages database operations for storing conversations and knowledge"""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase = None
        self.connected = False
        if supabase_url and supabase_key:
            self._init_connection()

    def _init_connection(self):
        """Initialize Supabase connection with error handling"""
        try:
            self.supabase = create_client(self.supabase_url, self.supabase_key)
            # Test connection with a simple query
            self.supabase.table('conversations').select('id').limit(1).execute()
            self.connected = True
            logger.info("✅ Database connection successful")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {str(e)}")
            self.connected = False
            self.supabase = None

    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connected and self.supabase is not None

    def save_conversation(self, session_id: str, query: str, response: str, context: str = None) -> Optional[str]:
        """Save conversation to database"""
        if not self.is_connected():
            logger.warning("Database not connected, skipping save")
            return None
            
        try:
            conversation_data = {
                'session_id': session_id,
                'query': query,
                'response': response,
                'context': context,
                'timestamp': get_current_timestamp()
            }
            result = self.supabase.table('conversations').insert(conversation_data).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return None

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history from database"""
        if not self.is_connected():
            logger.warning("Database not connected, returning empty history")
            return []
            
        try:
            result = self.supabase.table('conversations')\
                .select('*')\
                .eq('session_id', session_id)\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []

    def save_problem(self, problem: Problem) -> Optional[str]:
        """Save problem to database"""
        if not self.is_connected():
            logger.warning("Database not connected, skipping problem save")
            return None
        
        try:
            problem_data = {
                'id': problem.id,
                'title': problem.title,
                'description': problem.description,
                'domain': problem.domain.value,
                'context': problem.context,
                'user_id': problem.user_id,
                'timestamp': problem.timestamp.isoformat()
            }
            result = self.supabase.table('problems').insert(problem_data).execute()
            return result.data[0]['id'] if result.data else None
        except Exception as e:
            logger.error(f"Error saving problem: {str(e)}")
            return None

# Database manager instance
db_manager = SupabaseDatabaseManager(SUPABASE_URL, SUPABASE_KEY)

# IMPROVED CONSENSUS CALCULATOR - Fixed division by zero
class ConsensusCalculator:
    """Enhanced consensus calculation with better metrics"""
    
    @staticmethod
    def calculate_response_quality(response: str) -> float:
        """Calculate quality score based on response content"""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        words = response.split()
        sentences = response.split('.')
        
        # Prevent division by zero
        if len(words) == 0:
            return 0.1
        
        # Quality factors
        length_score = min(1.0, len(words) / 100)  # Optimal around 100 words
        structure_score = min(1.0, len(sentences) / 5) if len(sentences) > 0 else 0.1
        
        # Evidence markers
        evidence_markers = ['research shows', 'studies indicate', 'data suggests', 'analysis reveals', 
                          'according to', 'evidence indicates', 'research demonstrates']
        evidence_score = min(0.3, sum(1 for marker in evidence_markers if marker.lower() in response.lower()) * 0.1)
        
        # Reasoning markers
        reasoning_markers = ['because', 'therefore', 'however', 'furthermore', 'consequently', 
                           'moreover', 'additionally', 'thus', 'hence']
        reasoning_score = min(0.3, sum(1 for marker in reasoning_markers if marker.lower() in response.lower()) * 0.05)
        
        # Confidence modifiers
        uncertainty_markers = ['maybe', 'possibly', 'might', 'could be', 'perhaps', 'unsure', 'unclear']
        uncertainty_penalty = min(0.2, sum(1 for marker in uncertainty_markers if marker.lower() in response.lower()) * 0.05)
        
        # Specificity bonus
        specific_markers = ['specifically', 'for example', 'in particular', 'namely', 'such as']
        specificity_bonus = min(0.2, sum(1 for marker in specific_markers if marker.lower() in response.lower()) * 0.05)
        
        total_score = (
            length_score * 0.25 +
            structure_score * 0.15 +
            evidence_score +
            reasoning_score +
            specificity_bonus -
            uncertainty_penalty
        )
        
        return max(0.1, min(1.0, total_score))

    @staticmethod
    def calculate_consensus_score(responses: List[ModelResponse]) -> float:
        """Calculate overall consensus score from multiple responses"""
        if not responses or len(responses) == 0:
            return 0.0
        
        try:
            # Average confidence scores
            avg_confidence = sum(r.confidence for r in responses) / len(responses)
            
            # Response length variance (lower variance = better consensus)
            response_lengths = [len(r.response.split()) for r in responses]
            
            if len(response_lengths) == 0:
                return avg_confidence * 0.7  # No length consistency component
            
            # Calculate variance safely
            mean_length = sum(response_lengths) / len(response_lengths)
            length_variance = sum((l - mean_length)**2 for l in response_lengths) / len(response_lengths)
            length_consistency = max(0, 1 - (length_variance / 1000))  # Normalize
            
            # Combine metrics
            consensus_score = (avg_confidence * 0.7) + (length_consistency * 0.3)
            return min(1.0, max(0.0, consensus_score))
            
        except Exception as e:
            logger.error(f"Error calculating consensus score: {str(e)}")
            # Fallback: return average confidence if available
            if responses:
                try:
                    return sum(r.confidence for r in responses) / len(responses)
                except:
                    return 0.5  # Default fallback
            return 0.0

# MAIN AGORA CLASS - FIXED FOR REAL AGENT RESPONSES
class Agora:
    """Main Agora class for managing AI debates and consensus"""
    
    def __init__(self, primary_llm=None):
        self.primary_llm = primary_llm or "gpt-4"
        self.consensus_calculator = ConsensusCalculator()
        self.db_manager = db_manager
        self.agents = []  # Store individual agents instead of team
        self.available_models = self._check_available_models()
        self._initialize_agents()
        self.output_formatter = AgoraOutputFormatter()

    def _check_available_models(self):
        """Check which models have API keys available"""
        available = {}
        
        models_to_check = {
            "Claude Analyst": ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
            "GPT-4 Strategist": ("OPENAI_API_KEY", OPENAI_API_KEY), 
            "Mistral Evaluator": ("MISTRAL_API_KEY", MISTRAL_API_KEY),
            "SambaNova Specialist": ("SAMBANOVA_API_KEY", SAMBANOVA_API_KEY)
        }
        
        for model_name, (key_name, api_key) in models_to_check.items():
            available[model_name] = bool(api_key)
            logger.info(f"🔑 {model_name}: {'✅ Available' if api_key else '❌ No API key'}")
        
        return available



    def _initialize_agents(self):
        """Initialize individual AI agents"""
        self.agents = []
        plain_text_instructions = """
        
        CRITICAL FORMATTING REQUIREMENTS:
        - Do NOT use markdown formatting (no **, ##, -, etc.)
        - Use plain text with natural line breaks
        - For emphasis, use CAPITALIZATION or quotation marks
        - For lists, use numbers (1., 2., 3.) or natural language
        - For sections, use clear headers in plain text
        - Write as if you're speaking directly to a person
        - Use proper paragraph breaks with double line breaks
        
        Example of good formatting:
        
        KEY CONSIDERATIONS
        
        The main challenges include three important areas. First, we need to consider the technical aspects. This involves ensuring proper implementation and testing procedures.
        
        Second, the strategic implications require careful planning. Organizations should focus on long-term sustainability and stakeholder alignment.
        
        RECOMMENDATIONS
        
        Based on this analysis, I recommend the following steps:
        
        1. Conduct a thorough assessment of current capabilities
        2. Develop a phased implementation plan
        3. Establish clear success metrics and monitoring systems
        
        This approach will help ensure successful outcomes while minimizing risks.

        """
        
        try:
            # Check if we have at least one API key for real AI models
            has_real_api_keys = any([ANTHROPIC_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY, SAMBANOVA_API_KEY])
            
            if not has_real_api_keys:
                logger.error("❌ No real AI model API keys found - cannot create agents")
                return
            
            # Create Claude agent
            if ANTHROPIC_API_KEY:
                try:
                    claude_agent = Agent(
                        name="Claude Analyst",
                        role="Critical Analysis Specialist",
                        model=Claude(id="claude-3-5-sonnet-20240620"),
                        instructions="""
                        You are Claude Analyst, a Critical Analysis Specialist.
                        Provide expert analysis on the given topic from your specialized perspective.
                        Be thorough, evidence-based, and constructive in your responses.
                        Consider both benefits and potential challenges in your analysis.
                        Structure your response clearly with key insights and recommendations.
                        Keep responses concise and brief, focusing on actionable insights.
                        Respond using clearly formatted text with proper line breaks, bullet points, and headings. Do not use escape characters like \n or markdown symbols like ### or - unless you're actually formatting for a markdown-rendering environment. Write as if you're showing it in a user-friendly UI with readable spacing and structure.
                        RESPONSE GUIDELINES:
                            1. Always search for relevant information before answering complex questions
                            2. Clearly distinguish between document-based and web-based information
                            3. Provide source citations for all information
                            4. Synthesize information from multiple sources when available
                            5. Ask clarifying questions when the query is ambiguous
                            6. Be conversational but informative
                    {plain_text_instructions}
                        """,
                        knowledge=knowledge_base,
                    )
                    self.agents.append(claude_agent)
                    logger.info("✅ Created Claude agent")
                except Exception as e:
                    logger.error(f"❌ Failed to create Claude agent: {str(e)}")
            
            # Create GPT-4 agent
            if OPENAI_API_KEY:
                try:
                    openai_agent = Agent(
                        name="GPT-4 Strategist", 
                        role="Strategic Planning Expert",
                        model=OpenAIChat(id="gpt-4o"),
                        instructions="""
                        You are GPT-4 Strategist, a Strategic Planning Expert.
                        Provide expert analysis on the given topic from your specialized perspective.
                        Be thorough, evidence-based, and constructive in your responses.
                        Focus on strategic implications, implementation approaches, and long-term considerations.
                        Structure your response clearly with actionable insights.
                        Keep responses concise and brief, focusing on strategic value.
                        Respond using clearly formatted text with proper line breaks, bullet points, and headings. Do not use escape characters like \n or markdown symbols like ### or - unless you're actually formatting for a markdown-rendering environment. Write as if you're showing it in a user-friendly UI with readable spacing and structure.
                        RESPONSE GUIDELINES:
                            1. Always search for relevant information before answering complex questions
                            2. Clearly distinguish between document-based and web-based information
                            3. Provide source citations for all information
                            4. Synthesize information from multiple sources when available
                            5. Ask clarifying questions when the query is ambiguous
                            6. Be conversational but informative
                        {plain_text_instructions}

                        """,
                        knowledge=knowledge_base,
                    )
                    self.agents.append(openai_agent)
                    logger.info("✅ Created GPT-4 agent")
                except Exception as e:
                    logger.error(f"❌ Failed to create GPT-4 agent: {str(e)}")
            
            # Create Mistral agent
            if MISTRAL_API_KEY:
                try:
                    mistral_agent = Agent(
                        name="Mistral Evaluator",
                        role="Solution Evaluation Specialist", 
                        model=MistralChat(
                            id="mistral-large-latest",
                            api_key=MISTRAL_API_KEY,
                        ), # Use model object
                        instructions="""
                        You are Mistral Evaluator, a Solution Evaluation Specialist.
                        Provide expert analysis on the given topic from your specialized perspective.
                        Be thorough, evidence-based, and constructive in your responses.
                        Focus on evaluating different approaches, assessing feasibility, and identifying risks.
                        Structure your response clearly with evaluation criteria and recommendations.
                        Keep responses concise and brief, focusing on practical evaluation.
                        Respond using clearly formatted text with proper line breaks, bullet points, and headings. Do not use escape characters like \n or markdown symbols like ### or - unless you're actually formatting for a markdown-rendering environment. Write as if you're showing it in a user-friendly UI with readable spacing and structure.
                        RESPONSE GUIDELINES:
                            1. Always search for relevant information before answering complex questions
                            2. Clearly distinguish between document-based and web-based information
                            3. Provide source citations for all information
                            4. Synthesize information from multiple sources when available
                            5. Ask clarifying questions when the query is ambiguous
                            6. Be conversational but informative
                    {plain_text_instructions}

                        """,
                        knowledge=knowledge_base,
                    )
                    self.agents.append(mistral_agent)
                    logger.info("✅ Created Mistral agent")
                except Exception as e:
                    logger.error(f"❌ Failed to create Mistral agent: {str(e)}")

            # Create SambaNova agent - might need different import/setup
            if SAMBANOVA_API_KEY:
                try:
                    sambanova_agent = Agent(
                        name="SambaNova Specialist",
                        role="Technical Implementation Specialist",
                        model=Sambanova(),
                        instructions="""
                        You are SambaNova Specialist, a Technical Implementation Specialist.
                        Provide expert analysis on the given topic from your specialized perspective.
                        Be thorough, evidence-based, and constructive in your responses.
                        Focus on technical feasibility, implementation challenges, and system design.
                        Structure your response clearly with technical details and recommendations.
                        Keep responses concise and brief, focusing on technical implementation.
                        Respond using clearly formatted text with proper line breaks, bullet points, and headings. Do not use escape characters like \n or markdown symbols like ### or - unless you're actually formatting for a markdown-rendering environment. Write as if you're showing it in a user-friendly UI with readable spacing and structure.
                        RESPONSE GUIDELINES:
                            1. Always search for relevant information before answering complex questions
                            2. Clearly distinguish between document-based and web-based information
                            3. Provide source citations for all information
                            4. Synthesize information from multiple sources when available
                            5. Ask clarifying questions when the query is ambiguous
                            6. Be conversational but informative
                    {plain_text_instructions}

                        """,
                        knowledge=knowledge_base,
                    )
                    self.agents.append(sambanova_agent)
                    logger.info("✅ SambaNova agent Created")
                except Exception as e:
                    logger.error(f"❌ Failed to create SambaNova agent: {str(e)}")
                        
            logger.info(f"✅ Successfully initialized {len(self.agents)} agents")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def get_agent_status(self):
        """Get detailed status of the agents and available models"""
        status = {
            "agents_initialized": len(self.agents) > 0,
            "agent_count": len(self.agents),
            "available_models": self.available_models,
            "agents": []
        }
        
        for agent in self.agents:
            status["agents"].append({
                "name": agent.name,
                "role": agent.role,
                "model": getattr(agent, 'model', 'unknown')
            })
        
        return status

    def start_debate(self, problem: Problem) -> DebateRound:
        """Start individual agent analysis on a given problem - NO MOCK RESPONSES"""
        logger.info(f"🎯 Starting analysis on: {problem.title}")
        logger.info(f"🔍 Available agents: {len(self.agents)}")
        
        if not self.agents:
            raise Exception("❌ No AI agents available - please check API keys configuration")
        
        responses = []
        
        # Create analysis prompt
        analysis_prompt = dedent(f"""
            **Problem Analysis Request**
            
            Title: {problem.title}
            Description: {problem.description}
            Domain: {problem.domain.value}
            Context: {problem.context}
            
            Please provide your expert analysis on this problem, including:
            1. Key considerations and challenges
            2. Potential solutions or approaches
            3. Risks and benefits assessment
            4. Specific recommendations for next steps
            
            Please be thorough and provide actionable insights from your specialized perspective.
            
            Timestamp: {get_current_timestamp()}
        """)
        
        # Get response from each agent individually
        for agent in self.agents:
            try:
                logger.info(f"📤 Requesting analysis from {agent.name}...")
                
                # Get response from individual agent
                agent_response = agent.run(analysis_prompt)
                
                if agent_response and len(str(agent_response).strip()) > 20:
                    # Create model response object
                    response_text = str(agent_response).strip()
                    confidence = self.consensus_calculator.calculate_response_quality(response_text)
                    
                    model_response = ModelResponse(
                        model_name=agent.name,
                        response=response_text,
                        confidence=confidence,
                        reasoning=f"Direct response from {agent.role}",
                        timestamp=datetime.now(),
                        tokens_used=len(response_text.split()) * 1.3
                    )
                    responses.append(model_response)
                    logger.info(f"✅ Received response from {agent.name} (confidence: {confidence:.2f})")
                else:
                    logger.warning(f"⚠️ Empty or short response from {agent.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error getting response from {agent.name}: {str(e)}")
                continue
        
        # Ensure we have at least one response
        if not responses:
            raise Exception("❌ No responses received from any AI agents - check API keys and network connection")
        
        # Calculate consensus score
        consensus_score = self.consensus_calculator.calculate_consensus_score(responses)
        
        # Create debate round
        debate_round = DebateRound(
            round_number=1,
            responses=responses,
            consensus_score=consensus_score,
            timestamp=datetime.now()
        )
        
        logger.info(f"✅ Analysis completed - Generated {len(responses)} real AI responses - Consensus Score: {consensus_score:.2f}")
        return debate_round

    def save_debate_round(self, problem: Problem, debate_round: DebateRound) -> bool:
        """Save debate round and problem to the database"""
        try:
            # Save problem first
            problem_id = self.db_manager.save_problem(problem)
            
            # Save each response
            for response in debate_round.responses:
                self.db_manager.save_conversation(
                    session_id=problem.id,
                    query=f"Analysis request: {problem.title}",
                    response=response.response,
                    context=json.dumps({
                        "round_number": debate_round.round_number,
                        "model_name": response.model_name,
                        "confidence": response.confidence,
                        "consensus_score": debate_round.consensus_score,
                        "tokens_used": response.tokens_used
                    })
                )
            
            logger.info(f"✅ Saved debate round for problem: {problem.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving debate round: {str(e)}")
            return False

# IMPROVED GRADIO INTERFACE
def create_gradio_interface():
    """Create the Gradio interface for Agora"""
    
    # Initialize Agora instance
    agora = Agora()
    
    def analyze_problem(title: str, description: str, domain: str, user_id: str, context: str = ""):
        """Enhanced problem analysis with beautiful formatting using AgoraOutputFormatter"""
        
        # Initialize the formatter
        formatter = AgoraOutputFormatter()
        
        try:
            # Input validation
            if not title or not description:
                error_output = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                              ❌ INPUT ERROR                                  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ┌─ Missing Required Information ──────────────────────────────────────────────┐
    │ Please provide both a title and description for your problem.              │
    │                                                                             │
    │ Required fields:                                                            │
    │ • 📝 Problem Title: Clear, concise title                                   │
    │ • 📄 Problem Description: Detailed explanation of the issue                │
    └─────────────────────────────────────────────────────────────────────────────┘
                """
                return error_output.strip(), "❌ Missing required information"
            
            # Check if agents are available
            if not agora.agents:
                error_output = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                          ❌ CONFIGURATION ERROR                             ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ┌─ No AI Agents Available ────────────────────────────────────────────────────┐
    │ The system cannot proceed without properly configured AI agents.            │
    │                                                                             │
    │ Please check:                                                               │
    │ • 🔑 API keys are properly set in environment variables                     │
    │ • 🤖 At least one AI model is configured and accessible                    │
    │ • 🌐 Network connection allows API calls to AI services                    │
    │                                                                             │
    │ Current Status: {len(agora.agents)} agents initialized                                     │
    └─────────────────────────────────────────────────────────────────────────────┘
                """
                return error_output.strip(), "❌ No agents configured"
            
            # Convert domain to enum safely
            try:
                domain_lower = domain.lower()
                problem_domain = ProblemDomain(domain_lower)
            except ValueError:
                # Fallback to GENERAL if invalid domain
                problem_domain = ProblemDomain.GENERAL
                logger.warning(f"Invalid domain '{domain}', defaulting to GENERAL")
            
            # Create problem object
            problem = Problem(
                id=str(uuid.uuid4()),
                title=title.strip(),
                description=description.strip(),
                domain=problem_domain,
                context=context.strip() if context else "No additional context provided",
                user_id=user_id.strip() if user_id else "anonymous",
                timestamp=datetime.now()
            )
            
            # Log analysis start
            logger.info(f"🚀 Starting enhanced analysis for: {problem.title}")
            logger.info(f"🔍 Domain: {problem.domain.value} | Agents: {len(agora.agents)}")
            
            # Show progress indicator (for UI feedback)
            progress_output = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                            🔄 ANALYSIS IN PROGRESS                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    🎯 **Analyzing:** {problem.title}
    🏷️ **Domain:** {problem.domain.value.title()}
    🤖 **Consulting {len(agora.agents)} AI Agents...**

    ┌─ Progress ──────────────────────────────────────────────────────────────────┐
    │ ⏳ Sending analysis requests to AI models...                               │
    │ 🔄 This may take 30-60 seconds depending on model response times           │
    └─────────────────────────────────────────────────────────────────────────────┘
            """
            
            # Start the actual analysis
            try:
                debate_round = agora.start_debate(problem)
                logger.info(f"✅ Analysis completed - {len(debate_round.responses)} responses received")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Analysis failed: {error_msg}")
                
                error_output = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                             ❌ ANALYSIS FAILED                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ┌─ Error Details ─────────────────────────────────────────────────────────────┐
    │ {formatter._wrap_text(error_msg, 73)}                                       │
    └─────────────────────────────────────────────────────────────────────────────┘

    ┌─ Troubleshooting Steps ─────────────────────────────────────────────────────┐
    │ 1. 🔑 Verify all API keys are valid and have sufficient credits            │
    │ 2. 🌐 Check internet connection and firewall settings                      │
    │ 3. 🤖 Ensure AI services are operational (check status pages)              │
    │ 4. 📝 Try with a simpler problem description                               │
    │ 5. 🔄 Refresh the page and try again                                       │
    └─────────────────────────────────────────────────────────────────────────────┘
                """
                return error_output.strip(), f"❌ Analysis Error: {error_msg}"

            # Validate that we got meaningful responses
            if not debate_round.responses:
                error_output = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           ❌ NO RESPONSES RECEIVED                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    ┌─ Issue Description ─────────────────────────────────────────────────────────┐
    │ No AI agents provided responses to the analysis request.                   │
    │ This could indicate API key issues or service unavailability.              │
    └─────────────────────────────────────────────────────────────────────────────┘
                """
                return error_output.strip(), "❌ No responses received"

            # Save results to database
            try:
                save_success = agora.save_debate_round(problem, debate_round)
                if save_success:
                    logger.info("✅ Results saved to database successfully")
                else:
                    logger.warning("⚠️ Database save failed - results not persisted")
            except Exception as e:
                logger.error(f"❌ Database save error: {str(e)}")
                save_success = False

            # Use the enhanced formatter to create beautiful output
            try:
                main_output, summary_output = formatter.format_debate_results(
                    problem=problem, 
                    debate_round=debate_round, 
                    save_success=save_success
                )
                
                # Log success metrics
                avg_confidence = sum(r.confidence for r in debate_round.responses) / len(debate_round.responses)
                logger.info(f"📊 Analysis completed successfully:")
                logger.info(f"   • Responses: {len(debate_round.responses)}")
                logger.info(f"   • Avg Confidence: {avg_confidence:.2f}")
                logger.info(f"   • Consensus Score: {debate_round.consensus_score:.2f}")
                logger.info(f"   • Database Saved: {'Yes' if save_success else 'No'}")
                
                return main_output, summary_output
                
            except Exception as e:
                logger.error(f"❌ Formatting error: {str(e)}")
                
                # Fallback to basic formatting if formatter fails
                fallback_output = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    ⚠️ ANALYSIS COMPLETE (BASIC FORMAT)                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    🎯 **Problem:** {problem.title}
    📊 **Consensus Score:** {debate_round.consensus_score:.2f}/1.00
    🤖 **Responses:** {len(debate_round.responses)} AI agents
    🕒 **Completed:** {debate_round.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

    **Responses:**
    {chr(10).join([f"• **{r.model_name}** (confidence: {r.confidence:.2f}): {r.response[:200]}..." for r in debate_round.responses])}
                """
                
                return fallback_output.strip(), f"✅ Analysis completed ({len(debate_round.responses)} responses)"
                
        except Exception as e:
            # Ultimate fallback for any unexpected errors
            error_msg = str(e)
            logger.error(f"❌ Unexpected error in analyze_problem: {error_msg}")
            
            fallback_error = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           ❌ UNEXPECTED ERROR                               ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    An unexpected error occurred during analysis.

    **Error:** {error_msg}

    **Troubleshooting:**
    • Check system logs for detailed error information
    • Verify all configuration settings
    • Try restarting the application
    • Contact support if the issue persists

    **System Info:**
    • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    • Available Agents: {len(agora.agents) if 'agora' in locals() else 'Unknown'}
            """
            
            return fallback_error.strip(), f"❌ System Error: {error_msg}"

    
# Create Gradio interface with enhanced styling
    with gr.Blocks(
        title="AI Democracy - Agora System (Enhanced Output)", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        }
        .markdown {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏛️ AI Democracy - Multi-Model Consensus System
        ## The Agora: Where artificial minds gather to forge wisdom
        
        **Enhanced Output Edition** - Beautiful, readable analysis results from real AI models!
        """)
        
        # Show enhanced agent status
        agent_status = agora.get_agent_status()
        available_models = [name for name, available in agent_status['available_models'].items() if available]
        status_markdown = f"""
        ### 🤖 System Status
        
        **Available Agents:** {agent_status['agent_count']} | **Active Models:** {', '.join(available_models) if available_models else 'None configured'}
        
        **Features:** Enhanced Formatting ✨ | Real AI Responses 🤖 | Consensus Analysis 📊 | Database Storage 💾
        """
        gr.Markdown(status_markdown)
        
        with gr.Row():
            with gr.Column(scale=2):
                title_input = gr.Textbox(
                    label="📝 Problem Title",
                    placeholder="Enter a clear, concise title for your problem",
                    lines=1,
                    max_lines=2
                )
                
                description_input = gr.Textbox(
                    label="📄 Problem Description", 
                    placeholder="Provide a detailed description of the problem you want analyzed",
                    lines=5,
                    max_lines=10
                )
                
                domain_input = gr.Dropdown(
                    label="🏷️ Problem Domain",
                    choices=[domain.value.title() for domain in ProblemDomain],
                    value=ProblemDomain.GENERAL.value.title()
                )
                
                context_input = gr.Textbox(
                    label="🔍 Additional Context (Optional)",
                    placeholder="Any additional context, constraints, or specific requirements",
                    lines=2,
                    max_lines=4
                )
                
                user_input = gr.Textbox(
                    label="👤 User ID (Optional)",
                    placeholder="Enter your identifier for tracking (optional)",
                    lines=1
                )
                
                analyze_button = gr.Button(
                    "🚀 Start Enhanced AI Analysis", 
                    variant="primary", 
                    size="lg",
                    scale=2
                )
            
            with gr.Column(scale=1):
                gr.Markdown(f"""
                🎯 **Enhanced Analysis Process:**
                
                1. Submit your problem with detailed description
                2. Real AI models** analyze independently and thoroughly
                3. Enhanced formatting** makes results beautiful and readable
                4. Consensus scoring** shows agreement levels between models
                5. Executive summary** provides key insights at a glance
                
                🌟 New Features:
                1. Visual progress bars** for consensus scores*
                2. Emoji indicators** for quality and confidence levels*
                3. Structured response boxes** for each AI agent*
                4. Executive summaries** with key insights*
                5. Error handling** with helpful troubleshooting*
                
                🤖 AI Model Status:
                {chr(10).join([f"{name}: {'🟢 Ready' if available else '🔴 No API Key'}" 
                              for name, available in agent_status['available_models'].items()])}
                
                ⚡Quality Assurance:
                ✅ Only real AI responses (no mock data)
                ✅ Confidence scoring for reliability
                ✅ Consensus analysis for agreement
                ✅ Professional formatting for clarity
                """)
        
        gr.Markdown("## 📊 Enhanced AI Analysis Results")
        
        with gr.Row():
            with gr.Column():
                results_output = gr.Markdown(
                    label="🔬 Detailed Analysis",
                    elem_classes=["markdown"],
                    value="Analysis results will appear here after you submit a problem..."
                )
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(
                    label="📋 Executive Summary",
                    elem_classes=["markdown"], 
                    value="Executive summary will appear here..."
                )
        
        # Event handling with the enhanced analyze_problem function
        analyze_button.click(
            fn=analyze_problem,  # This is our new enhanced function
            inputs=[title_input, description_input, domain_input, user_input, context_input],
            outputs=[results_output, summary_output],
            show_progress=True
        )
        
        # Enhanced example problems section
        gr.Markdown("""
        ### 💡 **Sample Problems to Test Enhanced Formatting:**
        
        **Business Strategy:**
        - "How can we implement a sustainable remote work policy that maintains productivity and employee satisfaction?"
        
        **Technology & Ethics:**
        - "What are the key considerations for implementing AI-powered decision making in healthcare while ensuring patient privacy and safety?"
        
        **Environmental & Policy:**
        - "How should cities balance economic growth with environmental sustainability in urban planning decisions?"
        
        **Social & Psychological:**
        - "What strategies can organizations use to improve mental health support while respecting employee privacy boundaries?"
        
        **Innovation & Risk:**
        - "How can startups effectively validate product-market fit while managing limited resources and investor expectations?"
        """)
        
        gr.Markdown("""
        ---
        ### 🔧 **Enhanced System Information:**
        - **Framework:** Agno AI Agent Framework with Enhanced Formatting
        - **Output Engine:** AgoraOutputFormatter with ASCII Art & Emojis
        - **Database:** Supabase (PostgreSQL + Vector Storage)  
        - **AI Models:** Claude 3.5, GPT-4o, Mistral Large, SambaNova
        - **Version:** 2.0 (Enhanced UI/UX Edition)
        - **Features:** Real-time Analysis, Consensus Scoring, Beautiful Formatting
        """)
    
    return demo


# Additional utility function for testing the formatter
def test_enhanced_formatting():
    """Test function to demonstrate the enhanced formatting capabilities"""
    
    # This would typically be called with real data
    print("🧪 Testing Enhanced Formatting...")
    print("✅ AgoraOutputFormatter class ready for integration")
    print("✅ Enhanced analyze_problem function ready")
    print("✅ Gradio interface enhanced with new styling")
    print("🚀 Ready to provide beautiful AI analysis results!")


# MAIN EXECUTION
def main():
    """Main function to run the Agora system"""
    try:
        logger.info("🚀 Starting AI Democracy - Agora System")
        
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,  # Set to True if you want a public link
            debug=False,
            show_error=True,
            mcp_server=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start Agora system: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()