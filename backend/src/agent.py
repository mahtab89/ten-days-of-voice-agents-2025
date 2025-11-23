import logging
import json
import asyncio
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
# DAY 2: ORDER STATE
# ------------------------------

order_state = {
    "drinkType": None,
    "size": None,
    "milk": None,
    "extras": [],
    "name": None
}

# ------------------------------
# SAVE ORDER TOOL
# ------------------------------

@function_tool
async def save_order(context: RunContext, order: dict):
    """Save completed coffee order to JSON."""
    with open("final_order.json", "w") as f:
        json.dump(order, f, indent=4)
    return "Order saved successfully!"

# ------------------------------
# BARISTA AGENT
# ------------------------------

class BaristaAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly barista at BrewStar Café.

Your job:
- Take the customer’s voice order.
- Ask follow-up questions until ALL fields are filled:

{
  "drinkType": "",
  "size": "",
  "milk": "",
  "extras": [],
  "name": ""
}

Rules:
- Ask ONE question at a time.
- NEVER skip fields.
- Once all fields are collected, REPEAT the entire order to the customer for confirmation.
- After repeating the order, say:
  "Great! Your order is ready. I'm saving it now."
- Then call the save_order tool with the final order JSON.
- After saving, thank the customer and end politely.
""",
            tools=[save_order],
        )
        self.state = order_state

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

        llm=google.LLM(
            model="gemini-2.5-flash",
        ),

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

    # ------------------------------
    # Metrics
    # ------------------------------
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # ------------------------------
    # START SESSION
    # ------------------------------

    await session.start(
        agent=BaristaAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    print("🚀 Barista Agent is LIVE and ready to take orders!")
    await asyncio.sleep(1)
    await session.say("Welcome to BrewStar Café! What would you like to order today?")

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
