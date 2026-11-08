# Harness image. Build:  docker build -t hack:latest .
#
# * Base image tag is pinned (exact patch release, slim Debian 12).
# * Every Python dependency is pinned in requirements.txt (generated from requirements.in with
#   `uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu`, 159 packages).
# * --only-binary=:all: -> pip never compiles anything (slim image has no compiler, builds stay fast);
#   `uv pip compile --no-build` confirmed that every pinned package ships a wheel for CPython 3.12 / linux x86_64.
# * Everything is downloaded HERE, at build time. Nothing is fetched at run time (the container only reaches the
#   LLM proxy). No API keys are baked in: LLM_BASE_URL / LLM_API_KEY come from the environment of `docker run`.
FROM python:3.12.16-slim-bookworm

# The container runs with --read-only: do not try to write .pyc files at run time (they are compiled at install
# time by pip), and flush logs immediately so a hard kill at the time limit loses nothing.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --only-binary=:all: -r /app/requirements.txt

COPY *.py /app/

# The organisers override the entrypoint with `python` and pass the script path; these defaults match that call.
ENTRYPOINT ["python"]
CMD ["/app/run_agent.py", "--tasks-dir", "/data/tasks"]
