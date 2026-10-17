# Contrat local de propositions teacher

`yaaba.distillation_contract` crée un registre **text-free** de candidats à une
future expérience de distillation. Il ne charge aucun modèle, ne génère aucun
label et n'admet aucune ligne au train. Une ligne valide devient seulement
`eligible_for_review`; elle reste `label_origin=teacher_proposal`, jamais gold.

## Entrées

Le JSONL candidat contient uniquement des identités et des SHA-256 :

- source : `source_id`, tous les `group_ids` parent/page, SHA-256, split,
  provenance versionnée ;
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

La politique JSON doit explicitement fournir les splits permis, les splits
interdits, les groupes, identifiants et hashes de source interdits, ainsi que les
droits requis. `dev`, `val`, `validation`, `test` et `gold` doivent toujours
figurer parmi les splits interdits. Un split non explicitement permis est rejeté.

Exemple minimal de politique :

```json
{
  "schema_version": 1,
  "policy_id": "teacher-pool-v1",
  "allowed_splits": ["teacher_pool"],
  "forbidden_splits": ["dev", "val", "validation", "test", "gold"],
  "forbidden_group_ids": ["eval-document-001", "eval-page-002"],
  "forbidden_source_ids": ["heldout-item-001"],
  "forbidden_source_sha256": ["<sha256>"],
  "authorized_rights_statuses": ["authorized"],
  "required_rights_uses": ["teacher_generation", "model_training"]
}
```

Chaque candidat fournit les objets `source`, `teacher` et `generation`. La
source porte une liste non vide de `group_ids` afin de contrôler à la fois le
document parent et la page. La génération fournit `run_id` et les SHA-256 de son
manifeste, code, prompt, proposition et décodage. Le champ optionnel
`screening.reject_reasons` n'accepte que les défauts techniques listés plus
haut.

## Exécution CPU

```bash
python -m yaaba.distillation_contract \
  --input candidates.jsonl \
  --policy isolation-policy.json \
  --output teacher-proposal-registry.jsonl
```

Le registre est trié par `candidate_id`. Chaque décision et le manifeste ont un
SHA-256 canonique. Une répétition identique est sans effet ; l'outil refuse de
remplacer un registre différent. Le manifeste compagnon indique les volumes
`eligible_for_review` et `rejected`, et atteste que le registre ne contient pas
de texte, n'utilise pas de référence comme filtre et ne présente aucun label
teacher comme gold.

## Limites

Le contrôle par groupe dépend de la qualité des `group_ids` et de la liste
interdite fournie. Les SHA-256 détectent des identités exactes, pas les
paraphrases ni les traductions proches. Le contrat ne valide ni le mooré, ni la
fidélité, ni la qualité du teacher, ni la licence elle-même : il exige seulement
une déclaration de droits complète et autorisée par la politique. Toute
admission au SFT demande une étape séparée, un split d'évaluation indépendant et
soit une revue humaine compétente traçable, soit un bras teacher-only clairement
étiqueté.

Comme aucun texte brut n'entre dans cet outil, les SHA-256 de source et de
proposition sont des identités déclarées : un producteur amont doit les calculer
sur les octets convenus et les attester dans son propre manifeste. Ce registre
valide leur présence et leur forme, pas leur correspondance à un texte qu'il ne
lit volontairement pas.
