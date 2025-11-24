import logging
import json
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ------------------------------
# JSON FILE FOR LOGGING
# ------------------------------

LOG_FILE = "wellness_log.json"

def load_previous_logs():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_logs(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ------------------------------
# DAY 3 SAVE CHECK-IN TOOL
# ------------------------------

@function_tool
async def save_checkin(context: RunContext, mood: str, energy: str, goals: list, summary: str):
    """Save daily wellness check-in to JSON."""
    logs = load_previous_logs()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "mood": mood,
        "energy": energy,
        "goals": goals,
        "summary": summary
    }

    logs.append(entry)
    save_logs(logs)

    return "Daily check-in saved successfully."


# ------------------------------
# WELLNESS ASSISTANT
# ------------------------------

class WellnessAssistant(Agent):
    def __init__(self):
        previous_logs = load_previous_logs()

        last_entry = ""
        if previous_logs:
            last = previous_logs[-1]
            last_entry = f"Last time we talked, you said your mood was '{last['mood']}' and energy was '{last['energy']}'. "

        super().__init__(
            instructions=f"""
You are a gentle, supportive daily wellness companion.
You are NOT a doctor. Do NOT diagnose. Keep all advice simple and practical.

Start the conversation by greeting the user warmly.
Ask about:
1. Mood
2. Energy level
3. A few simple goals for today

If logs exist:
- Casually reference past data, for example:
  '{last_entry}'

After collecting:
- Provide a short simple reflection or encouragement.
- Then recap:
    - mood
    - energy
    - the 1–3 goals for today
- Ask: "Does this sound right?"

Then call the save_checkin tool with:
- mood
- energy
- goals (list)
- a short summary sentence

After saving, thank the user gently and end.
""",
            tools=[save_checkin]
        )


# ------------------------------
# PREWARM
# ------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ------------------------------
# ENTRYPOINT
# ------------------------------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = { "room": ctx.room.name }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start session
    await session.start(
        agent=WellnessAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    print("🌿 Wellness Companion is LIVE and listening...")

    await asyncio.sleep(1)
    await session.say("Hello! Let's take a moment to check in. How are you feeling today?")

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
