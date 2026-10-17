# Expériences locales SFT et évaluation sur Mac Apple Silicon

## Portée et identité des modèles

Ces scripts produisent des preuves techniques reproductibles. Ils ne valident
ni la qualité conversationnelle en mooré ni un modèle fondation. Une métrique
automatique ou un smoke synthétique n'est pas un jugement de locuteur.

Le vrai Yaaba n'était pas disponible dans le cache local le 17 octobre 2026.
Ses révisions distantes connues sont `6338b1d5ac936680701259f0f49c6035589c7563`
pour `burkimbia/yaaba-4B-base` et
`f94c7c87944b53a7decb2081d1be5c5804ced5d9` pour l'assistant. Un run sur
`McGill-NLP/AfriqueQwen3.5-4B-50Langs-Instruct-v1` révision
`0ff0ff64853f73e7065d6f743585309ad87755d3` est une nouvelle adaptation du
challenger AfriqueQwen3.5. Il ne constitue pas l'expérience E2 Yaaba.

## Environnement isolé

Le chemin principal est MLX-LM. Il exploite directement Apple Silicon et évite
une conversion lorsque le checkpoint Hugging Face est déjà chargeable. La voie
Transformers/PEFT sur MPS est un repli : elle consomme davantage de mémoire,
mais ses checkpoints restaurent aussi l'optimiseur et le scheduler.

```bash
python3.12 -m venv .venv-mac
.venv-mac/bin/python -m pip install -r requirements-mac-experiments.txt
.venv-mac/bin/python -m pytest -q
```

Les versions exercées sont MLX-LM 0.32.2, MLX 0.32.4, Transformers 5.20.1 et
Torch 2.14.1. MLX-LM 0.30.7 a introduit le support texte de Qwen3.5 ; 0.32.2
inclut les correctifs ultérieurs. Les API utilisées correspondent à la
[documentation LoRA officielle](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md),
au [config d'exemple officiel](https://github.com/ml-explore/mlx-lm/blob/main/examples/lora_config.yaml)
et à l'[API de génération officielle](https://github.com/ml-explore/mlx-lm#python-api).

## Contrats de sûreté

`train_mlx.py` impose :

- un fichier held-out réel pour tout entraînement ; son absence ou toute fuite
  bloque avant le chargement du modèle ;
- un split source conservé, ou un split SHA-256 déterministe par groupe ; un
  groupe présent dans deux splits bloque le run ;
- une expansion multi-tours en un préfixe par réponse assistant ; MLX-LM reçoit
  `mask_prompt: true`, et MPS vérifie token par token que le prompt est un
  préfixe exact avant de poser les labels `-100`. Un préflight avec le vrai
  tokenizer supprime les exemples tronqués et exige au moins un token assistant
  supervisé dans chaque exemple conservé. Le wrapper MLX impose aussi
  `enable_thinking: false` au chat complet et au préfixe de masque, puis refuse
  une frontière qui n'est pas un préfixe exact. Les messages utilisateur et le
  gabarit vide de raisonnement Qwen3 restent ainsi hors perte. Un tokenizer qui
  n'accepte pas cette option bloque le run au lieu de revenir silencieusement à
  son comportement par défaut ;
- des plafonds d'échantillons, pas, temps mural et mémoire. MLX fixe aussi la
  limite de son allocateur à 32 Gio par défaut ; un moniteur interrompt le groupe
  de processus au-dessus de cette limite ou après trois mesures sous 4 Gio de
  mémoire système disponible. MPS fixe sa fraction d'allocateur et vérifie les
  mêmes seuils aux frontières de pas ;
- un manifeste avec SHA-256 du corpus, du held-out, du template, du tokenizer et
  de tous les poids locaux ;
- un contrat de reprise immuable. Une modification de données, modèle, seed ou
  config oblige à ouvrir un autre dossier de run.

MLX-LM sauvegarde les adaptateurs périodiques et les recharge avec
`resume_adapter_file`. Cette reprise restaure les poids LoRA, mais pas l'état de
l'optimiseur, les gradients accumulés ni le curseur du chargeur ; le segment
suivant repart avec la seed de base augmentée du nombre de mises à jour déjà
terminées. Ce mécanisme est une nouvelle séquence déterministe depuis les poids
sauvegardés, pas une continuation bit-à-bit. Le backend MPS utilise les
checkpoints Trainer et restaure poids, optimiseur, scheduler et pas.

`--max-steps`, `--save-every`, `--eval-every` et `--report-every` sont toujours
exprimés en mises à jour de l'optimiseur. Pour MLX, le runner les multiplie par
`--grad-accumulation` avant de produire les micro-pas et refuse un checkpoint
hors frontière de mise à jour. Le contrat de reprise inclut le commit, l'état
Git et les versions du runtime. Une reprise trackée recharge obligatoirement le
Run ID MLflow persisté. Un échec ou toute limite de temps lève une erreur dans
le contexte de suivi et marque le segment `FAILED`, même si son checkpoint rend
la reprise possible. Un curseur de lignes empêche de relogguer les anciennes
métriques lors de cette reprise.

Le corpus `moore_assistant_sft.jsonl` ne fournit pas actuellement un identifiant
de document ou de groupe dont l'indépendance soit prouvée ; son champ `source`
grossier traverse 205 splits. Le runner refuse donc ce corpus par défaut. Le
drapeau `--allow-ungrouped-technical-pilot` conserve les splits source, marque
`grouping_verified: false` et limite la conclusion à la validation logicielle et
à la rétropropagation. Ce mode ne produit aucune mesure de qualité ni split gold.

Le pilote recommandé a été produit par le commit données
`ea0f56a1847886094277bd7d0b6f6785d5e5034a` : 512 lignes et 74 groupes fins,
avec zéro doublon, conflit ou correspondance held-out selon les contrôles exacts
et normalisés du pipeline. Son SHA-256 est
`094b32d4a9649f40f11f9674b80f8ae6a29a704ab760aa706f601c1f4394d0e1`.
Il reste explicitement `technical-only` et n'a pas reçu de validation
linguistique humaine.

Ce pilote ne contient pas de split validation indépendant. Le runner inscrit
donc `validation: disabled_no_independent_validation`, configure zéro batch de
validation et ne calcule aucune `validation_loss`. Il ne prélève jamais une
validation silencieuse dans les lignes d'entraînement.

## Template Qwen3.5 explicite

Le snapshot AfriqueQwen3.5 cité plus haut n'embarque pas de `chat_template`.
Le laisser implicite rendrait prompt et masque indéterminés. Le fichier
`configs/qwen3_5_chat_template.jinja` vient du tokenizer officiel
`Qwen/Qwen3.5-4B` à la révision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. La condition locale désactive le
thinking par défaut lorsque MLX-LM ne transmet pas le drapeau. Le runner crée
un overlay de symlinks et ne copie ni ne convertit les 8,4 Go de poids.

## Préparer et lancer un pilote borné

Les chemins ci-dessous illustrent le challenger public. Le corpus corrigé et le
held-out restent locaux et ne sont jamais recopiés dans Git.

```bash
MODEL="$HOME/.cache/huggingface/hub/models--McGill-NLP--AfriqueQwen3.5-4B-50Langs-Instruct-v1/snapshots/0ff0ff64853f73e7065d6f743585309ad87755d3"
BURKIMBIA_ROOT="/Users/josias/Documents/Personnal/BurkimbIA"
DATA="$BURKIMBIA_ROOT/ml-research/agents/data/worktrees/bia-datasets-text/output/yaaba-lineage/pilot.jsonl"
HELDOUT="$BURKIMBIA_ROOT/yaaba-4B/evaluation/tenu_a_lecart.jsonl"

.venv-mac/bin/python scripts/local/train_mlx.py \
  --backend mlx \
  --model "$MODEL" \
  --model-revision 0ff0ff64853f73e7065d6f743585309ad87755d3 \
  --chat-template configs/qwen3_5_chat_template.jinja \
  --data "$DATA" \
  --heldout "$HELDOUT" \
  --group-field lineage.group_hash \
  --technical-pilot \
  --output artifacts/runs/afriqueqwen35-pilot \
  --max-train-samples 128 \
  --max-valid-samples 32 \
  --max-test-samples 32 \
  --max-steps 5 \
  --walltime-minutes 10 \
  --max-seq-length 512 \
  --batch-size 1 \
  --grad-accumulation 1 \
  --num-layers 2 \
  --save-every 1 \
  --eval-every 2 \
  --report-every 1 \
  --memory-limit-gb 32 \
  --minimum-available-gb 4
```

Ajouter `--dry-run` prépare les splits, hash et commandes sans charger le
modèle. `--mock-smoke` exécute uniquement un fixture logiciel explicitement
marqué `TECHNICAL_MOCK`. Pour reprendre, relancer la même commande avec
`--resume`. Le plafond `--max-steps` est le total cumulé, pas le nombre de pas
supplémentaires.

Avec la branche MLOps intégrée, ajouter `--track` journalise manifeste,
métriques, ressources et checkpoints dans MLflow local
`http://127.0.0.1:5001`. Une reprise MLflow peut aussi fournir
`--mlflow-run-id`. Le manifeste autonome reste écrit même sans MLflow.

## Évaluation baseline et adaptée

L'évaluateur accepte des lignes `{"id": ..., "messages": [...]}` ou
`{"id": ..., "prompt": ...}`. Une référence optionnelle autorise exact match
normalisé et chrF. Les réponses, nombres de tokens, latences, fin de génération,
hash du prompt rendu et identité du décodage sont sauvegardés. Les prompts et
références gated ne sont pas dupliqués dans les sorties.

```bash
.venv-mac/bin/python scripts/local/evaluate_local.py \
  --backend mlx \
  --model "$MODEL" \
  --model-revision 0ff0ff64853f73e7065d6f743585309ad87755d3 \
  --chat-template configs/qwen3_5_chat_template.jinja \
  --questions /chemin/local/questions.jsonl \
  --output artifacts/eval/afriqueqwen35-base \
  --label afriqueqwen35-base \
  --max-samples 128 \
  --max-tokens 256 \
  --walltime-minutes 30
```

Pour l'adapté, conserver exactement questions, template, seed et décodage, puis
ajouter `--adapter /chemin/adapters`. Les deux manifestes permettent de vérifier
leur identité. Le décodage est greedy (`temperature=0`), sans draft model et
avec thinking désactivé. Le rendu du template est fail-closed : si le tokenizer
ne supporte pas `enable_thinking=False` ou si le renderer échoue, l'évaluation
s'arrête au lieu de retirer silencieusement l'option.

## Compatibilité observée du challenger

Le 17 octobre 2026, le snapshot AfriqueQwen3.5 ci-dessus a été chargé
directement par MLX-LM 0.32.2, sans conversion. Un prompt français greedy a
produit 25 tokens puis EOS : 2,52 s, 54,1 tokens/s de décodage, pic MLX 8,54 Go.
`/usr/bin/time -l` a observé 3,11 Go de RSS maximal et 9,25 Go de peak memory
footprint ; le processus complet, incluant le SHA-256 des 8,41 Go de poids, a
duré 11,08 s. Le SHA-256 du poids est
`55da672dc001dcc6a1515e699818dab94151a4d9d3d62bde6173a4b7ea62d342`.

Cela prouve le chargement et la génération texte locale pour cette révision.

Un pilote LoRA technique de trois mises à jour a ensuite utilisé 15 exemples du
pilote lineage à 256 tokens, une couche et un rank 4. Les pertes assistant-only
observées ont été 4,151, 3,070 et 3,311. Chaque checkpoint diffère du précédent
(239 615 puis 239 616 éléments modifiés ; normes L2 0,01528 et 0,01602), ce qui
confirme une rétropropagation et des mises à jour de poids. L'entraînement a duré
13,66 s ; le processus complet a culminé à 3,15 Go de RSS et le moniteur n'a pas
observé moins de 7,66 Go disponibles. Le run MLflow est
`584e8671382040eb8ca61c38ffa67b87`. Les métriques loss/tokens ont été reparsées
et journalisées après le run, car la table Rich de MLX-LM 0.32.2 différait du
format historique ; le correctif est couvert par un test de régression.

Ce pilote prouve que le chargement, le masque, la rétropropagation, les
checkpoints et le suivi tiennent dans ce budget précis. Il ne mesure pas la
qualité des réponses en mooré et ne remplace aucune évaluation humaine.

## Pilote préenregistré de 64 mises à jour

Le protocole `operations/2026-10-17-autonomous/PROTOCOLE-PILOTE.md` a été
exécuté sur les 28 prompts fixes, puis 64 mises à jour LoRA, puis les mêmes 28
prompts avec l'adaptateur. Les identités modèle, template, questions, ordre,
seed et décodage sont égales entre les deux évaluations ; seul l'adaptateur
diffère.

Le train a utilisé 440 exemples effectifs : 512 lignes source, 472 après les
exclusions directionnelles déclarées, puis 32 exemples rejetés parce que la
réponse ne tenait pas entièrement dans 256 tokens. Les 64 pertes et les quatre
checkpoints 16/32/48/64 sont enregistrés dans le run MLflow
`c1a55b8f21794c7393a5e5b58cfd4977`. Le train a duré 19,07 s ; le moniteur a
observé au plus 3,842 Go de RSS et au moins 8,355 Go disponibles. La moyenne des
huit premières pertes est 4,5028, celle des huit dernières 3,1724. Ce sont des
métriques d'optimisation bruitées, sans validation indépendante.

| Mesure automatique | Base | Adapté |
|---|---:|---:|
| Prompts terminés | 28 | 28 |
| Exact match normalisé, 24 références | 0 | 0 |
| chrF, 24 références | 3,6791 | 4,2802 |
| Tokens générés moyens | 64,9 | 52,7 |
| Générations au plafond de 128 | 7 | 3 |
| Latence moyenne | 1,360 s | 1,142 s |

Runs MLflow : baseline `c4df8b94059d4970b1264edc12e8feb8`, adapté
`15f30c1cf8304c82ac488d47af4e4e07`, analyse appariée posthoc
`66c2c45f6bef4912857fd2ca71591410`. Les SHA-256 des sorties sont respectivement
`07cca35a36430a57592f8886f2aa7e71fb19bb842f0e38ac5098f635cc3e345a` et
`dc4a1d71ee0007638bf6bf7f9890e5611faf5c951a45dad2b6f109a7fa95eae6`.

Le changement de chrF ne doit pas être interprété comme un gain de compétence.
Les prompts demandent une forme lexicale courte, alors que plusieurs sorties
sont des phrases explicatives : exact match et chrF mélangent accès lexical et
format de réponse. Un diagnostic posthoc de containment reste à zéro dans les
deux bras et les deux strates. Aucun score humain n'a été produit.

## Borne de loss MLX corrigée pour les nouveaux runs

MLX-LM 0.32.2 remplit ses tableaux de batch avec l'ID `0`, puis sa loss par
défaut utilise une borne supérieure inclusive (`steps <= length`). Elle inclut
ainsi une cible de remplissage après chaque séquence. Cet ID `0` n'est pas le
`pad_token_id` du tokenizer AfriqueQwen (`248044`) et se décode en `!`.

Le wrapper Mac injecte maintenant, uniquement pendant son propre processus,
une loss utilisant `steps < length`. Le masque conserve tous les tokens réels
de la réponse, dont `<|im_end|>` et la nouvelle ligne qui le suit, et exclut la
première position de remplissage. Un ID `0` qui apparaît réellement à
l'intérieur d'une réponse reste supervisé. La même fonction est transmise par
MLX à l'entraînement et à la validation. Le hook est retiré dans un bloc
`finally`, y compris après une exception.

Cette correction s'applique seulement aux nouveaux runs exécutés avec ce
wrapper. Les métriques historiques conservent le contrat MLX d'alors et ne sont
pas réécrites. Pour le pilote440 à masque assistant corrigé, l'audit daté compte
9 216 cibles de séquence réelles et 440 cibles de remplissage supplémentaires
dans la loss historique. Cela ne démontre pas que le remplissage explique les
générations atteignant leur plafond.
