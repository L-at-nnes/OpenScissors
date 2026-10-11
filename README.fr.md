# OpenScissors

**Détourez le sujet d'une photo, hors ligne, sur votre propre machine.** Une IA propose le contour, vous le corrigez au pinceau, puis vous posez le résultat sur un fond transparent, uni ou une image (avec une ombre si vous voulez) et vous exportez.

[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![Fait en Rust](https://img.shields.io/badge/fait%20en-Rust-b7410e)](https://www.rust-lang.org)
[![UI : Slint](https://img.shields.io/badge/UI-Slint-2379f4)](https://slint.dev)
[![Plateforme](https://img.shields.io/badge/plateforme-Windows-0078d4)](#obtenir-lappli)
[![Portable](https://img.shields.io/badge/installation-aucune%20(un%20dossier)-success)](#portable-pour-de-vrai)
[![Sans télémétrie](https://img.shields.io/badge/télémétrie-aucune-lightgrey)](#vie-privée)

*English: [README.md](README.md)*

<p align="center">
  <img src="docs/images/composed.png" alt="OpenScissors avec une tasse détourée posée sur un fond crème avec une ombre douce" width="900">
</p>

> Projet indépendant, sans lien avec un produit commercial. Il ne contient ni code, ni icônes, ni traductions, ni poids de modèle venant d'un tel produit.

## Pourquoi

Je voulais un outil de détourage qui

- fonctionne **hors ligne** (vos photos ne quittent jamais l'ordinateur),
- démarre instantanément et **ne mange pas la moitié de la RAM** pour rester ouvert,
- permet de **corriger** le résultat au lieu d'espérer que l'IA ait tout bon,
- tient dans **un seul dossier** à copier sur une clé USB.

Voilà. Environ 240 Mo de mémoire avec une photo ouverte (modèle non compté), 28 Mo au repos.

## Ce que ça fait

- **Détourage automatique** dès l'ouverture d'une image. Le résultat s'affiche à côté de l'original ; les deux vues zooment et se déplacent ensemble.
- **Correction au pinceau** : marquer ce qu'il faut garder, effacer ce qui dépasse, et un outil **cheveux** pour des bords doux.
- **Grouper par couleur** (traits verts et rouges) quand l'IA hésite : les zones de couleur proche suivent vos traits.
- **Sélection intelligente** : cliquez ce que vous voulez, clic droit ce que vous ne voulez pas, ou tracez un cadre (modèle SAM 2, léger).
- **Fonds** : transparent, couleur unie, une image, ou l'original avec le sujet comblé. Flou, niveaux de gris, négatif, pixellisation ou coloration sur l'une ou l'autre couche.
- **Ombre, recadrage à un ratio, marge**, puis export en PNG, JPEG, WebP, TIFF ou BMP. Les sources 16 bits peuvent rester en 16 bits.
- **Traitement par lot** d'un dossier entier avec vos réglages enregistrés.
- **4 langues** : English (par défaut), Français, Deutsch, Русский.
- **Plug-ins** (optionnels, désactivés tant que vous ne les activez pas) : de petits programmes lancés sur le fichier que vous venez d'enregistrer.

<p align="center">
  <img src="docs/images/smart-select.png" alt="Sélection intelligente sur la tasse" width="440">
  <img src="docs/images/models.png" alt="Le panneau des modèles" width="440">
</p>

## Obtenir l'appli

Windows 10/11, 64 bits.

1. Téléchargez le dernier `OpenScissors-…-portable-win-x64.zip` depuis la page Releases et décompressez-le où vous voulez.
2. Lancez `OpenScissors.exe`.
3. Ouvrez une image (le bouton, <kbd>Ctrl</kbd>+<kbd>O</kbd>, glisser-déposer, ou `OpenScissors.exe photo.jpg`).

Au premier lancement, aucun modèle d'IA n'est installé (ils ne sont pas fournis). Cliquez sur la pastille en haut, choisissez-en un, **Télécharger**. Le fichier est vérifié par SHA-256 avant usage.

| Modèle | Téléchargement | Mémoire | Pour |
|---|---|---|---|
| **IS-Net** (par défaut) | 179 Mo | ~0,6 Go | la plupart des photos, moins d'une seconde |
| BiRefNet lite | 224 Mo | ~5 Go | cheveux et structures fines, quelques secondes |
| BiRefNet lite, demi-précision | 115 Mo | ~5 Go | pareil, pensé pour les GPU |
| BiRefNet complet | 973 Mo | lourd | meilleure qualité, lent sur processeur |
| SAM 2.1 tiny | 156 Mo | ~1 Go pendant la sélection | l'outil Sélection intelligente |

L'appli prend le modèle le plus léger que vous avez tant que vous n'en choisissez pas un autre, et refuse un modèle qui ne tiendrait pas dans son budget mémoire plutôt que de figer votre PC.

## Portable pour de vrai

Tout ce que l'appli écrit reste dans son dossier :

```
OpenScissors/
├── OpenScissors.exe
├── openscissors-cli.exe   le même moteur dans un terminal
├── DirectML.dll        seulement pour le mode GPU optionnel
├── models/             ce que vous avez téléchargé
├── data/               réglages, position de la fenêtre, vos défauts
├── temp/               fichiers temporaires (nettoyés automatiquement)
└── licenses/
```

Supprimez `data/` pour repartir de zéro, `models/` pour libérer de la place. Rien n'est écrit dans le registre ni dans votre profil. (Si le dossier est en lecture seule, par exemple Program Files, l'appli se rabat sur votre profil plutôt que d'échouer.)

## Léger par défaut, lourd si vous le demandez

- Rien n'est chargé au lancement. Le modèle se charge avec votre première image.
- L'appli reste sous **25 % de votre RAM** (50 % dans les *Réglages*, si vous avez la mémoire et la patience pour BiRefNet).
- Les grandes photos sont traitées par bandes : une 8K coûte quelques centaines de Mo, pas des gigaoctets.
- La barre du bas affiche en direct la RAM et le CPU utilisés par l'appli.

## Raccourcis

| | |
|---|---|
| <kbd>F</kbd> garder · <kbd>B</kbd> supprimer (en *Grouper par couleur*) · <kbd>E</kbd> gomme | <kbd>S</kbd> sélection intelligente · <kbd>H</kbd> cheveux · <kbd>Maj</kbd>+<kbd>H</kbd> gomme cheveux |
| <kbd>M</kbd> déplacer · <kbd>Espace</kbd> maintenu pour faire glisser | <kbd>[</kbd> <kbd>]</kbd> taille du pinceau |
| <kbd>Ctrl</kbd>+<kbd>O</kbd> ouvrir · <kbd>Ctrl</kbd>+<kbd>S</kbd> enregistrer · <kbd>Ctrl</kbd>+<kbd>Maj</kbd>+<kbd>S</kbd> enregistrer sous | <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Y</kbd> annuler / rétablir |
| <kbd>Ctrl</kbd>+<kbd>0</kbd> 100 % · <kbd>Ctrl</kbd>+<kbd>9</kbd> ajuster · <kbd>Ctrl</kbd>+<kbd>+</kbd> / <kbd>-</kbd> zoom | molette : déplacer, <kbd>Ctrl</kbd>+molette : zoomer |
| <kbd>Entrée</kbd> appliquer une sélection · <kbd>Retour arrière</kbd> retirer le dernier clic · <kbd>Échap</kbd> annuler | |

Le pinceau est **désactivé par défaut** : l'appli démarre sur l'outil *Déplacer*, le curseur ne peint donc jamais par surprise.

## Vie privée

Aucun compte, aucune télémétrie, aucun appel réseau hormis les téléchargements de modèles que vous lancez vous-même (HTTPS, somme de contrôle vérifiée).

## Compiler soi-même

Il suffit de Rust (stable).

```bash
cargo run -p openscissors --release                 # lancer
cargo test --workspace                              # tests
node scripts/package-portable.mjs --build --zip     # le dossier portable et son zip, dans dist/
```

Les builds de debug gardent leurs données dans `portable-dev/`. `OPENSCISSORS_HOME` pointe l'appli vers un autre dossier, `OPENSCISSORS_MODEL=/chemin/model.onnx` charge un fichier local de type IS-Net.

Un outil en ligne de commande existe aussi : `cargo run -p openscissors-cli -- cut photo.jpg detourage.png` (et `predict`, `batch`, `models`).

Ajouter votre propre modèle : créez `models/<nom>/model.onnx` et un `model.json` à côté (voir l'exemple dans [README.md](README.md#using-your-own-model)).

## Comment c'est fait

- `crates/core` : tout ce qui n'est pas une fenêtre. Lecture/écriture d'images, exécution des modèles (ONNX Runtime), GrabCut et max-flow, matting, SAM 2, le document et son historique, rendu et export. Aucune dépendance graphique, donc facile à tester.
- `crates/app` : la fenêtre, écrite avec [Slint](https://slint.dev) et son moteur de rendu logiciel (≈ 100 Mo de gagnés face à un rendu GPU, beaucoup plus face à une vue web).
- `crates/cli` : la ligne de commande.

Détails dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) et [docs/PERF.md](docs/PERF.md) (mémoire et vitesse, mesurées).

## Limites connues

- Windows est la plateforme testée. Le code est portable, les builds Linux et macOS ne sont pas testés.
- Le mode GPU (DirectML) fonctionne mais n'a rien accéléré sur la carte essayée ; il reste optionnel.
- Le recadrage au clavier n'existe pas encore ; les poignées demandent un pointeur.
- Les messages d'erreur venant de Rust sont en anglais uniquement.
- Les traductions n'ont pas été relues par des locuteurs natifs. Les corrections sont les bienvenues.

## Licence

Apache-2.0. Voir [LICENSE](LICENSE), [NOTICE](NOTICE) et [THIRD_PARTY.md](THIRD_PARTY.md).

La bibliothèque d'interface, Slint, est utilisée sous la **Slint Royalty-free License 2.0** (gratuite pour les applications de bureau, avec l'attribution *Made with Slint* visible dans **Menu → À propos**). Slint existe aussi en GPL-3.0, ce qui rendrait un build d'OpenScissors GPL-3.0 dans son ensemble ; chacun choisit la licence qui lui convient en compilant les sources. Ceci n'est pas un avis juridique.

Les modèles sont des œuvres distinctes, avec leurs propres licences (MIT, Apache-2.0), listées dans [THIRD_PARTY.md](THIRD_PARTY.md).
