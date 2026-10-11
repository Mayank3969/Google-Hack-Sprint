# Model artifact placeholder

This directory is kept in the repository so the Docker build context has a
stable destination for model artifacts. The trained M4 checkpoint is produced
outside source control and must be copied here as:

```text
model/bigru/bigru_best.pt
```

The runtime Dockerfile requires that non-empty file and fails the image build
when only this placeholder is present.
