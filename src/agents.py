"""
Multi-Agent System - Specialized agents for different tasks
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from groq import Groq

from src.config import Config
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline, RAGResponse


class AgentRole(Enum):
    ROUTER = "router"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SUMMARIZER = "summarizer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    agent: str
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResult:
    """Final result from multi-agent processing"""
    final_answer: str
    query: str
    agent_trace: List[AgentMessage]
    sources: List[Dict[str, Any]]
    total_tokens: int
    latency_ms: int
    agents_used: List[str]


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, role: AgentRole, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.tokens_used = 0

    def _call_llm(self, user_message: str, context: str = "") -> str:
        """Call Groq LLM with agent's system prompt"""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nTask:\n{user_message}"})
        else:
            messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        self.tokens_used += response.usage.total_tokens
        return response.choices[0].message.content

    def run(self, query: str, context: str = "") -> AgentMessage:
        raise NotImplementedError


class RouterAgent(BaseAgent):
    """Determines which agents to invoke and in what order"""

    SYSTEM_PROMPT = """You are a routing agent. Analyze queries and return a JSON routing plan.
Return ONLY valid JSON in this format:
{
  "query_type": "factual|analytical|summary|comparison|multi_hop",
  "complexity": "simple|medium|complex",
  "agents_needed": ["researcher", "analyst", "synthesizer"],
  "search_queries": ["query1", "query2"],
  "reasoning": "brief explanation"
}"""

    def __init__(self):
        super().__init__("Router", AgentRole.ROUTER, self.SYSTEM_PROMPT)

    def run(self, query: str, context: str = "") -> AgentMessage:
        response = self._call_llm(f"Route this query: {query}")
        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            plan = {
                "query_type": "factual",
                "complexity": "simple",
                "agents_needed": ["researcher", "synthesizer"],
                "search_queries": [query],
                "reasoning": "Default routing"
            }

        return AgentMessage(
            agent=self.name,
            role=self.role,
            content=json.dumps(plan, indent=2),
            metadata=plan
        )


class ResearcherAgent(BaseAgent):
    """Retrieves and organizes relevant information"""

    SYSTEM_PROMPT = """You are a research agent. Your job is to:
1. Analyze retrieved document chunks
2. Identify the most relevant information
3. Organize findings clearly
4. Flag any gaps or contradictions in the sources
Be thorough and cite sources."""

    def __init__(self, vector_store: VectorStore):
        super().__init__("Researcher", AgentRole.RESEARCHER, self.SYSTEM_PROMPT)
        self.vector_store = vector_store

    def run(self, query: str, context: str = "") -> AgentMessage:
        results = self.vector_store.semantic_search(query, top_k=6)
        if not results:
            return AgentMessage(
                agent=self.name,
                role=self.role,
                content="No relevant documents found.",
                metadata={"results_count": 0}
            )

        raw_context = "\n\n".join(
            f"[{r.source} | score: {r.score:.2f}]\n{r.text}"
            for r in results
        )
        analysis = self._call_llm(
            f"Analyze these retrieved chunks for the query: '{query}'\nOrganize the key findings.",
            context=raw_context
        )

        return AgentMessage(
            agent=self.name,
            role=self.role,
            content=analysis,
            metadata={
                "results_count": len(results),
                "sources": [r.source for r in results],
                "top_score": results[0].score if results else 0,
                "raw_results": [{"text": r.text, "source": r.source, "score": r.score} for r in results]
            }
        )


class AnalystAgent(BaseAgent):
    """Performs deep analysis on retrieved information"""

    SYSTEM_PROMPT = """You are an analytical agent. Your job is to:
1. Identify patterns and insights in the research
2. Draw logical conclusions from evidence
3. Highlight implications and significance
4. Identify what's known vs uncertain
Be analytical, precise, and evidence-based."""

    def __init__(self):
        super().__init__("Analyst", AgentRole.ANALYST, self.SYSTEM_PROMPT)

    def run(self, query: str, context: str = "") -> AgentMessage:
        analysis = self._call_llm(
            f"Analyze the following research findings for the query: '{query}'",
            context=context
        )
        return AgentMessage(
            agent=self.name,
            role=self.role,
            content=analysis,
            metadata={"analysis_type": "deep_analysis"}
        )


class SummarizerAgent(BaseAgent):
    """Creates concise summaries of information"""

    SYSTEM_PROMPT = """You are a summarization agent. Your job is to:
1. Distill complex information into clear summaries
2. Preserve the most important points
3. Use bullet points for clarity
4. Maintain accuracy while being concise"""

    def __init__(self):
        super().__init__("Summarizer", AgentRole.SUMMARIZER, self.SYSTEM_PROMPT)

    def run(self, query: str, context: str = "") -> AgentMessage:
        summary = self._call_llm(
            f"Summarize the following information relevant to: '{query}'",
            context=context
        )
        return AgentMessage(
            agent=self.name,
            role=self.role,
            content=summary,
            metadata={"summary_type": "extractive"}
        )


class CriticAgent(BaseAgent):
    """Reviews and validates the analysis quality"""

    SYSTEM_PROMPT = """You are a critic agent. Your job is to:
1. Check if the analysis fully answers the query
2. Identify any logical errors or gaps
3. Suggest improvements if needed
4. Provide a quality score (1-10)
Be constructive and specific."""

    def __init__(self):
        super().__init__("Critic", AgentRole.CRITIC, self.SYSTEM_PROMPT)

    def run(self, query: str, context: str = "") -> AgentMessage:
        critique = self._call_llm(
            f"Critique this response for the query: '{query}'\nIs it accurate, complete, and helpful?",
            context=context
        )
        return AgentMessage(
            agent=self.name,
            role=self.role,
            content=critique,
            metadata={"critique_type": "quality_check"}
        )


class SynthesizerAgent(BaseAgent):
    """Combines all agent outputs into a final answer"""

    SYSTEM_PROMPT = """You are a synthesis agent. Your job is to:
1. Combine insights from multiple agents
2. Produce a coherent, comprehensive final answer
3. Cite sources clearly with [Source: name] format
4. Structure the response for maximum clarity
Produce the best possible final answer."""

    def __init__(self):
        super().__init__("Synthesizer", AgentRole.SYNTHESIZER, self.SYSTEM_PROMPT)

    def run(self, query: str, context: str = "") -> AgentMessage:
        final_answer = self._call_llm(
            f"Synthesize a final comprehensive answer for: '{query}'",
            context=context
        )
        return AgentMessage(
            agent=self.name,
            role=self.role,
            content=final_answer,
            metadata={"synthesis_type": "multi_agent"}
        )


class MultiAgentSystem:
    """Orchestrates multiple specialized agents"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.router = RouterAgent()
        self.researcher = ResearcherAgent(self.vector_store)
        self.analyst = AnalystAgent()
        self.summarizer = SummarizerAgent()
        self.critic = CriticAgent()
        self.synthesizer = SynthesizerAgent()

        self.agent_map = {
            "researcher": self.researcher,
            "analyst": self.analyst,
            "summarizer": self.summarizer,
            "critic": self.critic,
            "synthesizer": self.synthesizer,
        }

        self.run_history: List[AgentResult] = []

    def run(self, query: str, verbose: bool = True) -> AgentResult:
        """Run the full multi-agent pipeline"""
        start_time = time.time()
        trace: List[AgentMessage] = []

        if verbose:
            print(f"\n🤖 Multi-Agent System Starting...")
            print(f"   Query: {query[:80]}...")

        # Step 1: Router decides the plan
        if verbose:
            print(f"\n[1/4] 🗺️  Router analyzing query...")
        router_msg = self.router.run(query)
        trace.append(router_msg)
        plan = router_msg.metadata

        agents_to_run = plan.get("agents_needed", ["researcher", "synthesizer"])
        if "synthesizer" not in agents_to_run:
            agents_to_run.append("synthesizer")

        if verbose:
            print(f"   Plan: {plan.get('query_type')} | Agents: {agents_to_run}")

        # Step 2: Research phase
        if verbose:
            print(f"\n[2/4] 🔍 Researcher retrieving information...")
        research_msg = self.researcher.run(query)
        trace.append(research_msg)
        accumulated_context = f"RESEARCH:\n{research_msg.content}"

        # Step 3: Run remaining agents in sequence
        step = 3
        for agent_name in agents_to_run:
            if agent_name in ("researcher", "synthesizer"):
                continue
            agent = self.agent_map.get(agent_name)
            if not agent:
                continue

            if verbose:
                print(f"\n[{step}/4] ⚙️  {agent.name} processing...")

            msg = agent.run(query, context=accumulated_context)
            trace.append(msg)
            accumulated_context += f"\n\n{agent_name.upper()} OUTPUT:\n{msg.content}"
            step += 1

        # Step 4: Synthesizer produces final answer
        if verbose:
            print(f"\n[4/4] ✨ Synthesizer generating final answer...")
        final_msg = self.synthesizer.run(query, context=accumulated_context)
        trace.append(final_msg)

        # Collect sources from researcher
        sources = []
        if research_msg.metadata.get("raw_results"):
            seen = set()
            for r in research_msg.metadata["raw_results"]:
                if r["source"] not in seen:
                    seen.add(r["source"])
                    sources.append({
                        "source": r["source"],
                        "relevance_score": round(r["score"], 3),
                        "preview": r["text"][:120] + "..."
                    })

        total_tokens = sum(
            a.tokens_used for a in [
                self.router, self.researcher, self.analyst,
                self.summarizer, self.critic, self.synthesizer
            ]
        )

        result = AgentResult(
            final_answer=final_msg.content,
            query=query,
            agent_trace=trace,
            sources=sources,
            total_tokens=total_tokens,
            latency_ms=int((time.time() - start_time) * 1000),
            agents_used=[m.agent for m in trace]
        )

        self.run_history.append(result)

        if verbose:
            print(f"\n✅ Done in {result.latency_ms}ms | Agents: {result.agents_used}")

        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_runs": len(self.run_history),
            "avg_latency_ms": int(
                sum(r.latency_ms for r in self.run_history) / max(len(self.run_history), 1)
            ),
            "total_tokens": sum(r.total_tokens for r in self.run_history),
            "agents_available": list(self.agent_map.keys())
        }