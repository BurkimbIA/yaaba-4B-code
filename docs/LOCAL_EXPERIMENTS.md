# Expériences locales suivies avec MLflow

Ce module trace des exécutions locales reproductibles sans modifier les recettes
d'entraînement. Le manifeste reste utilisable sans MLflow; le serveur sert ensuite
à conserver les statuts, courbes, artefacts et checkpoints dans un même catalogue.

## Installation isolée

Utiliser Python 3.11 ou plus récent et un environnement réservé au suivi :

```bash
python3.11 -m venv .runtime/venv
.runtime/venv/bin/pip install -r requirements-tracking.txt
```

Le projet principal ne dépend pas de MLflow. `build_manifest` et
`write_manifest` fonctionnent donc même si le paquet n'est pas installé. Une
tentative d'ouvrir `tracked_run` sans MLflow lève `TrackingUnavailable` avec la
commande d'installation attendue.

## Serveur local persistant

Le répertoire d'état est obligatoire afin d'éviter une base créée implicitement
dans le dossier courant. Le script reste au premier plan; l'arrêter avec Ctrl-C.

```bash
.runtime/venv/bin/python scripts/local/start_mlflow.py \
  --state-dir "$HOME/Library/Application Support/BurkimbIA/mlflow-local"
```

La commande fixe l'écoute à `127.0.0.1:5001`, utilise un seul worker, écrit les
métadonnées dans `mlflow.db` et les artefacts dans `artifacts/` sous le dossier
d'état. Elle n'accepte aucun paramètre permettant une exposition sur le réseau.
La vue locale est disponible à <http://127.0.0.1:5001>.

## Contrat d'un run

```python
from pathlib import Path

from yaaba.experiment_tracking import build_manifest, tracked_run, write_manifest

manifest = build_manifest(
    run_kind="mlx-lora-smoke",
    evidence_kind="OBSERVED",
    config=Path("runs/pilot/resolved_config.json"),
    data=Path("runs/pilot/split_manifest.json"),
    code=Path("scripts/local/run_mlx_lora.py"),
    model={
        "uri": "McGill-NLP/AfriqueQwen3.5-4B-50Langs-Instruct-v1",
        "revision": "0ff0ff64853f73e7065d6f743585309ad87755d3",
        "sha256": "<sha256-du-contenu-certifié>",
    },
    tokenizer=Path("runs/pilot/tokenizer"),
    seed=20261017,
    budget={"max_steps": 2, "device": "cpu"},
    stop_criteria={"max_steps": 2},
)
write_manifest(manifest, "runs/pilot/manifest.json")

with tracked_run(
    manifest,
    experiment_name="yaaba-local-pilots",
    run_name="mlx-lora-smoke-001",
) as run:
    # Conserver run.run_id avec l'état local du runner pour une reprise explicite.
    run.log_metric("wall_seconds", 4.2, kind="technical", step=1)
    run.log_metric("dev/loss", 1.7, kind="ml", step=1)
    run.log_artifact("runs/pilot/resolved_config.json", artifact_path="config")
    run.log_checkpoint("runs/pilot/checkpoint-1", step=1,
                       metadata={"resumable": True})
```

Les chemins locaux sont hachés par contenu. Une ressource distante est passée
sous forme de mapping avec son SHA-256 certifié; sans SHA-256 explicite, le
module étiquette honnêtement l'empreinte comme empreinte des métadonnées
canoniques. Le manifeste complet a sa propre empreinte. MLflow la conserve comme
paramètre immuable et comme artefact `run-manifest/manifest.json`.

Pour reprendre un run, reconstruire ou relire exactement le même manifeste et
passer son identifiant :

```python
with tracked_run(manifest, experiment_name="yaaba-local-pilots",
                 run_name="nom-ignoré-à-la-reprise", run_id=saved_run_id) as run:
    ...
```

La reprise vérifie le paramètre et l'artefact avant de remettre le statut à
`RUNNING`. Un manifeste différent lève `ManifestMismatch` sans modifier le run.
Une sortie normale produit `FINISHED`; toute exception produit `FAILED`, note
seulement le type de l'exception puis relance celle-ci. Son message n'est pas
copié car il pourrait contenir un secret.

## Interprétation des preuves et métriques

`evidence_kind` vaut `OBSERVED`, `HISTORICAL`, `DECLARED`, `HYPOTHESIS` ou
`PROPOSED`. Importer une courbe ancienne dans un run `HISTORICAL` distinct; ne
pas la mélanger à un run fraîchement exécuté. Un test synthétique ou un smoke
test CPU est `OBSERVED`, mais ses métriques restent techniques et ne deviennent
pas une mesure de qualité du modèle.

Chaque métrique doit choisir son espace :

- `kind="technical"` pour durée, mémoire, débit, reprise et succès logiciel;
- `kind="ml"` pour loss, chrF, WER ou jugement humain obtenu par le protocole.

MLflow les stocke respectivement sous `technical/...` et `ml/...`. Le module
rejette les valeurs NaN/infinies, les steps négatifs et les métriques sans type.

Les mappings du manifeste et des métadonnées d'artefacts masquent les clés de
type mot de passe, token, clé API ou secret. Le module ne lit jamais
l'environnement. Il ne peut pas certifier le contenu d'un fichier arbitraire :
ne jamais lui transmettre un `.env`, une configuration contenant des
identifiants, des logs bruts de headers ou un checkpoint embarquant des secrets.

## Validation locale

La suite utilise un stockage temporaire si aucun serveur n'est fourni :

```bash
.runtime/venv/bin/pytest tests/test_experiment_tracking.py
```

Pour vérifier l'API HTTP, démarrer un serveur temporaire puis exécuter :

```bash
YAABA_TEST_MLFLOW_URI=http://127.0.0.1:5001 \
  .runtime/venv/bin/pytest tests/test_experiment_tracking.py
```

Ce test crée uniquement de petits fichiers texte synthétiques. Il vérifie le
statut `RUNNING` puis `FINISHED`, la reprise, la terminaison `FAILED`, le refus
d'un manifeste différent, les deux espaces de métriques et les artefacts
adressés par contenu. Il ne charge aucun poids et ne mesure aucune qualité ML.
