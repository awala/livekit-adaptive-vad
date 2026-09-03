import inspect
import logging
from typing import Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

InterruptionMode = Literal["adaptive", "vad"]
AGENT_SESSION_SUPPORTS_TURN_HANDLING = (
    "turn_handling" in inspect.signature(AgentSession).parameters
)

ASSISTANT_INSTRUCTIONS = """You are a passionate, talkative voice assistant who loves explaining things in detail. You speak in long, flowing explanations and always aim for at least five to eight sentences per response. You enjoy going on tangents and adding color commentary.

When asked about any topic, give a thorough explanation. Do not rush. Take your time. Add examples, analogies, and asides. If the user asks a simple question, still give a rich, detailed answer.

You are interacting via voice. Rules:
Respond in plain text only. No markdown, lists, JSON, code, or emojis.
Spell out numbers and URLs naturally.
Avoid acronyms.

If the user interrupts you, acknowledge it naturally and pivot to what they are saying. Do not apologize for being cut off. Just flow with the conversation like a human would.

If the user says tell me about yourself or say something long, launch into a detailed monologue about any interesting topic such as history, science, cooking, or travel so they have something easy to talk over."""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=ASSISTANT_INSTRUCTIONS)


class SessionEntrypoint:
    def __init__(
        self,
        *,
        agent_name: str,
        interruption_mode: InterruptionMode,
    ) -> None:
        self.agent_name = agent_name
        self.interruption_mode = interruption_mode

    async def __call__(self, ctx: JobContext) -> None:
        await run_session(
            ctx,
            agent_name=self.agent_name,
            interruption_mode=self.interruption_mode,
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


def _build_turn_handling(interruption_mode: InterruptionMode):
    if not AGENT_SESSION_SUPPORTS_TURN_HANDLING:
        raise RuntimeError(
            "The installed LiveKit SDK does not expose AgentSession(turn_handling=...). "
            "Install livekit-agents>=1.5 before running this comparison."
        )

    return TurnHandlingOptions(
        # Both variants still rely on VAD to notice overlapping speech.
        # Adaptive mode adds LiveKit's classifier on top of that VAD signal.
        turn_detection=MultilingualModel(),
        interruption={
            "mode": interruption_mode,
        },
    )


def build_session(ctx: JobContext, interruption_mode: InterruptionMode) -> AgentSession:
    return AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=_build_turn_handling(interruption_mode),
        preemptive_generation=True,
    )


async def run_session(
    ctx: JobContext,
    *,
    agent_name: str,
    interruption_mode: InterruptionMode,
) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "interruption_mode": interruption_mode,
    }

    logger.info(
        "Starting %s with interruption.mode=%s",
        agent_name,
        interruption_mode,
    )

    session = build_session(ctx, interruption_mode)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await ctx.connect()


def create_server(
    *,
    agent_name: str,
    interruption_mode: InterruptionMode,
    port: int | None = None,
) -> AgentServer:
    server = AgentServer(port=port) if port is not None else AgentServer()
    server.setup_fnc = prewarm
    server.rtc_session(
        SessionEntrypoint(
            agent_name=agent_name,
            interruption_mode=interruption_mode,
        ),
        agent_name=agent_name,
    )

    return server
