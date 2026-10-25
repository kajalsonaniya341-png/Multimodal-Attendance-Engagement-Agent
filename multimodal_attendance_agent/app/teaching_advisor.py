"""
AI Teaching Suggestion Engine Module
EPIC 5: Teaching Suggestion Engine & Pedagogical Advisory
Privacy-First Design: 100% Offline Heuristic Advisory Agent
Platform ID: 6ac775fb32e2e6be34c17468
"""

import time
from typing import Dict, List, Optional, Any
import pandas as pd


class PedagogicalAdvisory:
    """Represents a discrete pedagogical teaching suggestion."""
    def __init__(
        self,
        timestamp_sec: float,
        level: str,  # "Action Required", "Timely Tip", "Classroom Dynamic", "Positive Affirmation"
        trigger: str,
        recommendation: str,
        pedagogical_rationale: str
    ):
        self.timestamp_sec = timestamp_sec
        self.timestamp_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"
        self.level = level
        self.trigger = trigger
        self.recommendation = recommendation
        self.pedagogical_rationale = pedagogical_rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_str": self.timestamp_str,
            "timestamp_sec": self.timestamp_sec,
            "level": self.level,
            "trigger": self.trigger,
            "recommendation": self.recommendation,
            "rationale": self.pedagogical_rationale
        }


class TeachingAdvisorEngine:
    """
    Offline Heuristic AI Advisory Agent generating real-time, non-intrusive pedagogical prompts
    when classroom collective attention flags, distraction spikes, or monologue fatigue occurs.
    """

    def __init__(self, lull_threshold: float = 45.0, high_threshold: float = 70.0):
        self.lull_threshold = lull_threshold
        self.high_threshold = high_threshold
        self.cooldown_sec = 15.0  # Prevent flooding prompts
        self.last_trigger_times: Dict[str, float] = {}

    def _can_trigger(self, rule_key: str, current_time: float) -> bool:
        last = self.last_trigger_times.get(rule_key, -999.0)
        if (current_time - last) >= self.cooldown_sec:
            self.last_trigger_times[rule_key] = current_time
            return True
        return False

    def evaluate_telemetry_stream(self, telemetry_df: pd.DataFrame) -> List[PedagogicalAdvisory]:
        """
        Scans synchronized telemetry stream and derives temporal advisories.
        """
        if telemetry_df.empty or "fused_engagement_index" not in telemetry_df.columns:
            return []

        advisories: List[PedagogicalAdvisory] = []
        self.last_trigger_times.clear()

        window_size = 5  # 5-step rolling window
        lecture_streak = 0

        for i in range(len(telemetry_df)):
            row = telemetry_df.iloc[i]
            t = row["timestamp_sec"]
            score = row["fused_engagement_index"]
            ac_state = row.get("acoustic_state", "")
            vis_score = row.get("visual_attentiveness", 70.0)

            # Track continuous monologue lecture
            if ac_state == "Instructor Lecture":
                lecture_streak += 1
            else:
                lecture_streak = 0

            # Window statistics
            start_idx = max(0, i - window_size + 1)
            recent_scores = telemetry_df.iloc[start_idx:i + 1]["fused_engagement_index"]
            mean_recent = float(recent_scores.mean())

            # Rule 1: Collective Attention Lull (< 45% for sustained window)
            if mean_recent < self.lull_threshold and self._can_trigger("lull", t):
                advisories.append(PedagogicalAdvisory(
                    timestamp_sec=t,
                    level="Action Required",
                    trigger=f"Collective engagement dropped below {self.lull_threshold}% (recent mean: {round(mean_recent, 1)}%)",
                    recommendation="Launch a quick 60-second interactive pulse poll, a think-pair-share prompt, or initiate a 2-minute physical stretch break.",
                    pedagogical_rationale="Cognitive attention naturally decays after 15–20 minutes of continuous reception. A low-stakes interactive retrieval task re-activates working memory without disrupting lesson pacing."
                ))

            # Rule 2: High Ambient Classroom Distraction
            elif ac_state == "High Ambient Classroom Distraction" and vis_score < 60.0 and self._can_trigger("distraction", t):
                advisories.append(PedagogicalAdvisory(
                    timestamp_sec=t,
                    level="Classroom Dynamic",
                    trigger="Elevated ambient chatter and acoustic disruption detected while forward gaze is fragmented",
                    recommendation="Execute an auditory refocusing cue (rhythmic clap or chime), pause speaking until room quiets, or clarify task instructions if students are confused.",
                    pedagogical_rationale="Unstructured peer chatter often signals ambiguity in task requirements. Reframing the immediate task goal restores collective focus faster than vocal reprimands."
                ))

            # Rule 3: Extended Monologue / Passive Lecture Fatigue
            elif lecture_streak >= 25 and self._can_trigger("monologue", t):
                advisories.append(PedagogicalAdvisory(
                    timestamp_sec=t,
                    level="Timely Tip",
                    trigger=f"Extended instructional monologue detected ({lecture_streak * 2}s continuous instructor talk without student dialogue)",
                    recommendation="Pivot instructional mode: pose an open concept check question, invite a student to rephrase the core definition, or open a 2-minute Q&A window.",
                    pedagogical_rationale="Passive listening without generative participation accelerates drowsiness and decreases long-term conceptual retention."
                ))

            # Rule 4: Silent Disengagement / Potential Confusion
            elif ac_state == "Silence" and vis_score < 50.0 and self._can_trigger("confusion", t):
                advisories.append(PedagogicalAdvisory(
                    timestamp_sec=t,
                    level="Classroom Dynamic",
                    trigger="Prolonged classroom silence paired with downward head poses and slumping postures",
                    recommendation="Conduct an immediate diagnostic check: 'Thumbs up if this concept makes sense, sideways if you need another example.'",
                    pedagogical_rationale="Students frequently go completely quiet when experiencing cognitive overload or embarrassment regarding conceptual difficulty."
                ))

            # Rule 5: Optimal Collaborative Synergy
            elif score >= self.high_threshold and ac_state == "Student Discussion / Verbal Interaction" and self._can_trigger("synergy", t):
                advisories.append(PedagogicalAdvisory(
                    timestamp_sec=t,
                    level="Positive Affirmation",
                    trigger=f"High collective engagement ({round(score, 1)}%) with active peer-to-peer verbal interaction",
                    recommendation="Active collaborative synergy achieved. Allow groups 2 more minutes to conclude discussions, then synthesize key student insights on the board.",
                    pedagogical_rationale="Peer verbalization reinforces cognitive schema construction and peer-to-peer scaffolding."
                ))

        # Always guarantee at least one initial pedagogical guidance if none fired
        if not advisories:
            advisories.append(PedagogicalAdvisory(
                timestamp_sec=0.0,
                level="Timely Tip",
                trigger="Session initialization & baseline tracking established",
                recommendation="Begin with a brief learning objective hook to frame today's core competency.",
                pedagogical_rationale="Priming student schema at the start of class increases sustained attention by up to 28%."
            ))

        return advisories

    def answer_pedagogical_query(self, user_question: str, current_session_summary: Optional[Dict[str, Any]] = None) -> str:
        """
        Simulated offline pedagogical assistant responding to educator queries.
        """
        q = user_question.lower()
        if "poll" in q or "quick poll" in q:
            return (
                "**Pedagogical Tip for Quick Pulse Polls:**\n"
                "1. **Multiple-Choice Concept Check:** Pose a single question with 3 options where 2 represent common misconceptions.\n"
                "2. **Physical Finger Voting:** Ask students to hold up 1, 2, or 3 fingers against their chest simultaneously.\n"
                "3. **Peer Justification:** If split, give students 60 seconds to convince their neighbor before re-polling."
            )
        elif "distract" in q or "noise" in q or "chatter" in q:
            return (
                "**Addressing Classroom Noise Non-Intrusively:**\n"
                "1. **Strategic Pause:** Stop talking mid-sentence and make calm eye contact; silence is far louder than raising your voice.\n"
                "2. **Auditory Cue:** Use a two-tap rhythm on the lectern or a soft chime.\n"
                "3. **Check for Blockers:** Quiet chatter often means 'What slide are we on?' Clarify the prompt in writing on screen."
            )
        elif "sleep" in q or "slump" in q or "tired" in q or "fatigue" in q:
            return (
                "**Re-energizing Drowsy / Slumping Students:**\n"
                "1. **Kinesthetic Reset:** 'Stand up, stretch towards the ceiling, turn to your partner and share one key takeaway.'\n"
                "2. **Low-Stakes Cold Call:** Call on a student with a low-pressure open question: 'Alex, in your own words, what did we just calculate?'\n"
                "3. **Change Modality:** Transition from lecture slides to a live problem demonstration on the whiteboard."
            )
        elif "discussion" in q or "group" in q:
            return (
                "**Optimizing Group Verbal Discussions:**\n"
                "1. **Assign Concrete Roles:** Designate a Spokesperson, Timekeeper, and Scribe for each pod.\n"
                "2. **Set a Strict Timer:** Display a visible countdown (e.g. 3 minutes); urgency increases focus.\n"
                "3. **Harvesting Insights:** Ask Group B to build upon or critique Group A's proposed solution."
            )
        else:
            return (
                f"**Pedagogical Advisory Insight:**\n"
                f"Based on current telemetry (Engagement Index: {current_session_summary.get('mean_engagement', 72.0) if current_session_summary else 72.0}%), "
                "the key to sustained cognitive attention is dynamic pacing. Alternate between 10-12 minutes of direct instruction and 3-5 minutes of active student retrieval."
            )
