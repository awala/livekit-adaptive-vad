from livekit.agents import cli

from agent_common import create_server

server = create_server(
    agent_name="interruption-adaptive",
    interruption_mode="adaptive",
    port=8082,
)


if __name__ == "__main__":
    cli.run_app(server)
