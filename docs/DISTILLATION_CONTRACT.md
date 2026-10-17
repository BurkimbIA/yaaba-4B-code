# Contrat local de propositions teacher

`yaaba.distillation_contract` crée un registre sans champ de contenu brut pour
une future expérience de distillation. Il ne charge aucun modèle, ne génère
aucun label et n'admet aucune ligne au train. Une ligne valide devient seulement
`eligible_for_review`; elle reste `label_origin=teacher_proposal`, jamais gold.

## Entrées

Le JSONL candidat contient uniquement des identités et des SHA-256 :

- source : `source_id`, `source_parent_ids`, `group_ids` typés, SHA-256, split,
  provenance épinglée à une révision immuable ;
- droits : statut, base, licence et usages autorisés ;
- teacher : modèle, révision, SHA-256 modèle et tokenizer ;
- génération déjà effectuée ailleurs : run, manifeste, code, prompt, proposition
  et décodage, tous identifiés par SHA-256 ;
- éventuels motifs de rejet techniques préenregistrés.

Le contrat rejette les champs inconnus. Il ne conserve donc ni source brute, ni
proposition brute, ni référence, ni gold. Les motifs automatiques admis portent
sur la mécanique de génération (`empty_proposal`, `truncated`, `source_copy`,
répétition, caractères, longueur, artefact de prompt ou doublon). Un score contre
une référence, un jugement de langue ou une préférence linguistique ne peuvent
pas devenir un filtre.

La politique JSON fournit les splits permis, les groupes, identifiants et hashes
de source interdits, les rôles de groupe requis, les datasets autorisés et les
contrats exacts de génération. Les alias connus de développement, validation,
test, évaluation, gold, heldout, benchmark et revue humaine sont interdits de
manière immuable dans le code. Ils ne peuvent pas être réautorisés par la
politique. Un split non explicitement permis est également rejeté.

Exemple minimal de politique :

```json
{
  "schema_version": 1,
  "policy_id": "teacher-pool-v1",
  "allowed_splits": ["teacher_pool"],
  "forbidden_splits": [],
  "forbidden_group_ids": ["eval-document-001", "eval-page-002"],
  "forbidden_source_ids": ["heldout-item-001"],
  "forbidden_source_sha256": ["<sha256>"],
  "required_group_kinds": ["document", "page"],
  "allowed_datasets": [
    {"dataset_id": "private-pool", "dataset_revision": "<40-or-64-hex>"}
  ],
  "allowed_generation_contracts": [
    {
      "contract_id": "teacher-run-1",
      "teacher": {
        "model_id": "org/model",
        "revision": "<40-or-64-hex>",
        "model_sha256": "<sha256>",
        "tokenizer_sha256": "<sha256>"
      },
      "generation": {
        "run_id": "run-001",
        "manifest_sha256": "<sha256>",
        "code_sha256": "<sha256>",
        "prompt_sha256": "<sha256>",
        "decoding_sha256": "<sha256>"
      }
    }
  ],
  "authorized_rights_statuses": ["authorized"],
  "required_rights_uses": ["teacher_generation", "model_training"]
}
```

Chaque candidat fournit les objets `source`, `teacher` et `generation`. La
source porte une liste non vide `source_parent_ids` et une table `group_ids` dont tous
les rôles requis par la politique sont présents. La génération nomme son
`contract_id`, son `run_id` et les SHA-256 du manifeste, du code, du prompt, de
la proposition et du décodage. L'identité teacher et l'identité de run doivent
être octet-identiques au contrat autorisé. Plusieurs teachers restent possibles
si chacun possède un contrat explicite ; le manifeste agrège les identités
réellement observées. Le champ optionnel `screening.reject_reasons` n'accepte que
les défauts techniques listés plus haut.

Les droits `teacher_generation` et `model_training` sont un plancher immuable,
même si la politique demande d'autres usages. Les révisions teacher et dataset
doivent être des identités de contenu hexadécimales de 40 ou 64 caractères ;
`main`, `latest` ou un tag mobile sont refusés.

## Exécution CPU

```bash
python -m yaaba.distillation_contract \
  --input candidates.jsonl \
  --policy isolation-policy.json \
  --output teacher-proposal-registry.jsonl
```

Le registre est trié par `candidate_id`. Chaque décision et le manifeste ont un
SHA-256 canonique. Une répétition identique est sans effet ; chaque fichier est
créé par lien exclusif et un writer concurrent ne peut pas remplacer un contenu
différent. Le JSONL et son manifeste sont toutefois deux fichiers séparés : un
arrêt machine entre les deux créations peut laisser un registre sans manifeste,
qui doit être considéré incomplet. Le manifeste compagnon indique les volumes
`eligible_for_review` et `rejected`, les identités dataset/teacher/run observées,
l'absence de champs source/proposition/référence, et le fait qu'aucun label
teacher n'est présenté comme gold.

## Limites

Le contrôle par groupe dépend de l'exhaustivité déclarée des `source_parent_ids`
et `group_ids`, ainsi que de la liste interdite fournie. L'outil impose les rôles
déclarés mais ne peut pas prouver qu'un producteur amont a fourni tous les vrais
parents. Les SHA-256 détectent des identités exactes, pas les
paraphrases ni les traductions proches. Le contrat ne valide ni le mooré, ni la
fidélité, ni la qualité du teacher, ni la licence elle-même : il exige seulement
une déclaration de droits complète et autorisée par la politique. Toute
admission au SFT demande une étape séparée, un split d'évaluation indépendant et
soit une revue humaine compétente traçable, soit un bras teacher-only clairement
étiqueté.

Les identifiants sont limités à 128 caractères ASCII sans espace et les champs
inconnus sont rejetés ; cela empêche de passer directement une phrase, mais ne
constitue ni une anonymisation ni un détecteur sémantique de texte. Comme aucun
champ de contenu brut n'entre dans cet outil, les SHA-256 de source et de
proposition sont des identités déclarées : un producteur amont doit les calculer
sur les octets convenus et les attester dans son propre manifeste. Ce registre
valide leur présence et leur forme, pas leur correspondance à un texte qu'il ne
lit volontairement pas.

Au 17 octobre 2026, l'inventaire Data ne fournit aucun pool immédiatement
admissible : les 7 686 lignes SFT techniques restantes n'ont pas encore de
droits joints par parent de source. Elles doivent donc rester rejetées avec
droits inconnus jusqu'à cette jointure. DEV32, heldout, lexiques, questions,
probes et leurs sorties modèle associées restent des exclusions fermes.
