# ☁️ GPU & Cloud Infrastructure

Terraform configurations for spinning up **GPU-enabled environments** to run deep learning notebooks (Phases 5–9). Designed for **quick launch and destroy** to minimize cost.

> **Note**: This folder will be fully populated in **Phase 5** (Deep Learning: Foundations). The Terraform modules below are placeholders showing what's coming.

---

## Planned Infrastructure

### Google Cloud Platform (GCP)

| Module | Purpose | Estimated Cost |
|--------|---------|----------------|
| `gcp/gke-gpu/` | GKE cluster with GPU node pool (T4/L4) | ~$0.35/hr per T4 GPU |
| `gcp/vertex-ai/` | Vertex AI Workbench notebook instance | ~$0.50/hr (n1-standard-4 + T4) |
| `gcp/compute-gpu/` | Single GPU VM (Compute Engine) | ~$0.35/hr per T4 GPU |

### Amazon Web Services (AWS)

| Module | Purpose | Estimated Cost |
|--------|---------|----------------|
| `aws/eks-gpu/` | EKS cluster with GPU node group | ~$0.50/hr per T4 GPU |
| `aws/sagemaker/` | SageMaker notebook instance | ~$0.50/hr (ml.g4dn.xlarge) |
| `aws/ec2-gpu/` | Single GPU EC2 instance | ~$0.50/hr (g4dn.xlarge) |

### Free / Low-Cost Alternatives

| Option | GPU | Time Limit | Cost |
|--------|-----|-----------|------|
| **Google Colab** | T4 (15 GB) | ~12 hr/session | Free |
| **Kaggle Notebooks** | T4 / P100 | 30 hr/week | Free |
| **Lightning AI** | T4 | 22 hr/month | Free |
| **Colab Pro** | A100 / V100 | Priority access | ~$10/month |

---

## Planned Directory Structure

```
infrastructure/
├── README.md                    ← You are here
├── variables.tf                 # Shared variables (region, project, etc.)
│
├── gcp/
│   ├── gke-gpu/
│   │   ├── main.tf             # GKE cluster + GPU node pool
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars.example
│   ├── vertex-ai/
│   │   └── ...
│   └── compute-gpu/
│       └── ...
│
├── aws/
│   ├── eks-gpu/
│   │   ├── main.tf             # EKS cluster + GPU node group
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── terraform.tfvars.example
│   ├── sagemaker/
│   │   └── ...
│   └── ec2-gpu/
│       └── ...
│
├── scripts/
│   ├── setup-gpu-drivers.sh    # NVIDIA driver + CUDA setup
│   ├── launch.sh               # Quick-start: terraform init + apply
│   └── destroy.sh              # Quick-teardown: terraform destroy
│
└── colab/
    └── colab_setup.ipynb       # Notebook for Colab GPU verification
```

---

## Quick Start (Coming in Phase 5)

```bash
# Launch a GCP GPU VM in ~5 minutes
cd infrastructure/gcp/compute-gpu
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project/region

terraform init
terraform apply

# When done — DESTROY to stop billing!
terraform destroy
```

---

## Design Principles

1. **Quick launch, quick destroy** — Every config includes a `destroy.sh` for cost safety
2. **Minimal cost** — Default to cheapest GPU options (T4 over V100/A100)
3. **Configurable** — GPU type, region, instance size all parameterized
4. **Preinstalled** — Startup scripts install PyTorch, CUDA, Jupyter automatically
5. **Safe** — `terraform.tfvars` is gitignored to prevent credential leaks

---

## When Is This Needed?

| Phase | GPU Required? | Notes |
|-------|--------------|-------|
| 1–4 | ❌ No | All CPU-compatible with small datasets |
| 5 | ⚡ Optional | MLP, basic CNN/RNN work on CPU; GPU speeds up training |
| 6 | ⚡ Recommended | Transformers, GNNs benefit from GPU |
| 7 | ⚡ Optional | RL environments are mostly CPU-bound |
| 8 | ⚡ Recommended | Self-supervised pretraining benefits from GPU |
| 9 | ✅ Recommended | Fine-tuning HuggingFace models strongly benefits from GPU |
| 10 | ❌ No | PGMs are CPU-based |
