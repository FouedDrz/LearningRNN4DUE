# LearningRNN4DUE
Ce dépôt contient un exemple de code Arduino pour implémenter un réseau de neurones récurrent (RNN) simple. Il peut être utilisé pour apprendre à partir de données séquentielles et effectuer des prédictions de sortie. Le code montre également comment effectuer la mise à jour des poids et des biais du réseau en utilisant la rétropropagation à travers le temps (BPTT).

## Fonctionnalités

- Implémentation d'un RNN simple avec une couche cachée
- Utilisation de la fonction d'activation sigmoïde
- Initialisation des poids et des biais en utilisant la méthode de Xavier/Glorot
- Mise à jour des poids et des biais avec la rétropropagation à travers le temps (BPTT)

## Prérequis

- Arduino IDE
- Une carte compatible Arduino

## Installation

1. Clonez ce dépôt ou téléchargez le code source en tant que fichier ZIP.
2. Ouvrez le fichier `learning_rnn.ino` dans l'IDE Arduino.
3. Modifiez les paramètres (taille d'entrée, taille cachée, taille de sortie, taux d'apprentissage) et les données d'entraînement selon vos besoins.
4. Téléversez le code sur votre carte Arduino.

## Utilisation

Le code comprend une fonction `rnn_forward()` pour effectuer une passe avant dans le réseau, et une fonction `rnn_backward()` pour effectuer la mise à jour des poids et des biais en utilisant la rétropropagation à travers le temps (BPTT). Les erreurs avant et après la mise à jour sont affichées sur le moniteur série.

Dans la boucle principale (`loop()`), le RNN est entraîné avec un exemple d'entrée et de sortie attendue. Vous pouvez modifier les exemples d'entrée et de sortie pour entraîner le RNN sur vos propres données.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
