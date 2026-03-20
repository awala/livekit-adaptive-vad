from livekit.agents import cli

from agent_common import create_server

server = create_server(
    agent_name="interruption-vad",
    interruption_mode="vad",
)


if __name__ == "__main__":
    cli.run_app(server)
