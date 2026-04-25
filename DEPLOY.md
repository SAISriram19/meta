# Deployment

## HF Spaces (recommended)

HF Spaces reads the YAML frontmatter in `README.md` automatically. All that's
left is creating the Space and pushing.

```bash
# One-time: create the Space on HF. Replace <username>.
# Visit https://huggingface.co/new-space, pick "Docker" template.
# Name it, for example: meta

# Then from the local repo:
git init -b main
git add .
git commit -m "initial meta drop"
git remote add hf https://huggingface.co/spaces/<username>/meta
git push hf main
```

HF Spaces builds the Dockerfile automatically. Build logs appear in the Space UI.
Once the container passes its healthcheck, the env is live at:
`https://<username>-meta.hf.space`

## Smoke-check the deployed env

```bash
SPACE=https://<username>-meta.hf.space

curl $SPACE/health
curl $SPACE/tasks
curl -X POST $SPACE/reset -H 'content-type: application/json' \
     -d '{"task_id":"L2_strategic_shift"}'
```

## Local Docker (smoke before pushing)

```bash
docker build -t meta .
docker run --rm -p 7860:7860 meta

# in another shell
curl localhost:7860/health
```

## openenv.yaml

`openenv.yaml` is shipped at the repo root for OpenEnv-registry submissions.
The hosted Space URL above is what the hackathon judges will hit.

## Pre-deploy checklist

1. `python tests/test_integration.py` — passes locally  
2. `python tests/test_long_horizon.py` — passes locally  
3. `README.md` frontmatter has correct `sdk: docker` and `app_port: 7860`  
4. `Dockerfile` builds without heavy optional deps (sentence-transformers is NOT
   required; memory falls back to a deterministic hash-embed)  
5. `.dockerignore` keeps the container slim (excludes eval outputs, notebooks, tests)

## Troubleshooting

- **Build OOM**: HF Spaces free tier has 16GB limit. Our container is under 500MB.
- **Cold-start slow**: first request initializes the scenario registry (~100ms). Subsequent requests reuse in-process state.
- **Episode state lost between requests**: the server is single-episode-per-process by design. Each `/reset` starts fresh.
- **Ground truth leaking**: should never happen — `Pydantic Field(exclude=True)` strips `ground_truth_tag` and `manipulation_pattern` from every serialization path. Tests lock this.
