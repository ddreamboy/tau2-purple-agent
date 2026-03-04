# Tau2 Green Agent

A green agent for the [Tau2 benchmark](https://github.com/sierra-research/tau2-bench) on the [AgentBeats](https://agentbeats.dev) platform. Evaluates purple agents on customer service tasks across multiple domains (airline, retail, telecom) using simulated users and real tool environments.

## How it works

The green agent runs tau2 evaluations via the [A2A protocol](https://a2a-protocol.org/latest/):

1. Receives an evaluation request with a purple agent URL and config (domain, number of tasks, etc.)
2. For each task, creates a simulated user and orchestrates a multi-turn conversation between the user, the purple agent, and the domain environment (tools, databases, policies)
3. Evaluates whether the purple agent completed the task successfully
4. Returns pass rate, per-task rewards, and timing metrics

## Project Structure

```
src/
├─ server.py      # A2A server setup and agent card
├─ executor.py    # A2A request handling
├─ agent.py       # Tau2 evaluation logic and RemoteA2AAgent wrapper
└─ messenger.py   # A2A messaging utilities
amber/
├─ amber-scenario.json5         # Amber scenario (green + purple + gateway)
├─ amber-manifest-green.json5   # Green agent manifest
├─ amber-manifest-purple.json5  # Purple agent manifest
├─ sample.env                   # Environment variable template
└─ README.md                    # Amber compile and run instructions
tests/
└─ test_agent.py  # A2A conformance tests
setup.sh          # Downloads tau2-bench data for local development
test_run.py       # Example evaluation request script
Dockerfile        # Docker image (includes tau2-bench data)
```

## Running Locally

```bash
# Clone tau2-bench data
bash setup.sh
export TAU2_DATA_DIR=$PWD/tau2-bench/data

# Install dependencies
uv sync

# Set API key for the UserSimulator LLM
export OPENAI_API_KEY=sk-...
# Or for Gemini:
# export GEMINI_API_KEY=...

# Start the green agent
uv run src/server.py
```

The server starts on port 9009. You'll need a purple agent running separately (e.g. from [agent-template](https://github.com/RDI-Foundation/agent-template)) to send evaluation requests to.

## Running with Docker

The Docker image bundles tau2-bench data, so no setup script is needed.

```bash
docker build -t tau2-green .
docker run -p 8081:8081 -e OPENAI_API_KEY=sk-... tau2-green
```

## Running with Amber

See [amber/README.md](amber/README.md) for instructions on compiling and running the full scenario (green agent + purple agent + gateway) using the Amber CLI.

## Configuration

The following config parameters can be passed in the evaluation request (or via Amber's `assessment_config`):

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `domain` | yes | `airline` | `airline`, `retail`, `telecom`, or `mock` |
| `num_tasks` | no | all | Limit number of tasks to run |
| `task_ids` | no | all | Specific task IDs to run |
| `max_steps` | no | `200` | Max orchestrator steps per task |
| `user_llm` | no | `openai/gpt-4o-mini` | LLM for the UserSimulator (litellm format) |
| `user_llm_args` | no | `{"temperature": 0.0}` | LLM arguments for the UserSimulator |

To run the full benchmark, submit one evaluation per domain.

## Testing

```bash
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The CI workflow builds, tests, and publishes to GitHub Container Registry on push to `main` or version tags:

```
ghcr.io/rdi-foundation/tau2-agentbeats:latest
ghcr.io/rdi-foundation/tau2-agentbeats:1.0.0
```
