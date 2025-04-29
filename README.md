# Strock_SciAdv_2025

Code from the paper: Strock, A., Mistry, P.K., Menon, V. (2025). Personalized deep neural networks reveal mechanisms of math learning disabilities in children. Science Advances.

In this project we use artificial neural networks to model Mathematical Learning Disabilities (MLD) in children as resulting from an higher excitability level of neurons.
We use the structure [CORnet](https://github.com/dicarlolab/CORnet) without any pre-training, and we train it to solve addition and subtraction visually presented.

## Initial installation


Make a new environment and install packages from `requirements.txt`, e.g. `pip install -r requirements.txt`.

Clone external repositories.
```bash
mkdir common/packages
cd common/packages
git clone https://github.com/dicarlolab/CORnet
git clone https://github.com/schung039/neural_manifolds_replicaMFT
```

## Setting up environment

```bash
source environment.sh
```

## Generation of the addition/subtraction dataset

```bash
submit8c dataset/addsub18_handwritten.py
```

```bash
submit8c dataset/addsub18_font.py
```

```bash
submit1c dataset/behavior_addsub18.py
```

## Model

Training the model
```bash
submit1g "model/train.py --scale 1.0"
```

Testing the model
```bash
submit1g "model/test.py --scale 1.0 --saveall --step $(seq -s ' ' 0 100 3800)"
```

Representational similarity analysis
```bash
submit8c analysis/similarity_analysis/addsub_similarity.py  --time 1-00:00:00
```

Behavioral analysis
```bash
submit8c analysis/behavioral_analysis/numberline_entropy.py
```

Manifold analysis
```bash
submit8c analysis/manifold_analysis/step_manifold.py --psteps $(seq -s ' ' 0 100 3800) --time 2-00:00:00 -p owners,normal --pmax 20 --mem 20G -c 32
```

# Manuscript Figures

To obtain all Figures of manuscript:
```bash
bash paper/figure_all.sh
```

To obtain Figure X of manuscript
```bash
python paper/figureX.py
```

To obtain Figure SI X of manuscript

```bash
python paper/figureSX.py
```