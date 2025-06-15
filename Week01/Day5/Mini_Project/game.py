# game.py

# On importe le module random pour générer un choix aléatoire
import random

# Définition de la classe Game qui gère le jeu
class Game:

    # Cette méthode demande à l'utilisateur de faire un choix valide
    def get_user_item(self):
        """Demande à l'utilisateur de choisir rock, paper ou scissors."""
        while True:  # Boucle jusqu'à ce que l'utilisateur entre une réponse valide
            user_input = input("Choose rock, paper or scissors: ").lower()  # On récupère l'entrée en minuscule
            if user_input in ["rock", "paper", "scissors"]:  # Si le choix est valide
                return user_input  # On retourne ce choix
            print("Invalid choice. Please try again.")  # Sinon, on affiche un message d'erreur

    # Cette méthode génère un choix aléatoire pour l'ordinateur
    def get_computer_item(self):
        """Fait un choix aléatoire pour l'ordinateur."""
        return random.choice(["rock", "paper", "scissors"])  # On utilise random.choice pour tirer un choix

    # Cette méthode compare les choix du joueur et de l'ordinateur pour déterminer le résultat
    def get_game_result(self, user_item, computer_item):
        """Détermine le résultat du jeu entre l'utilisateur et l'ordinateur."""
        if user_item == computer_item:  # Cas d'égalité
            return "draw"
        elif (
            (user_item == "rock" and computer_item == "scissors") or  # Le joueur gagne
            (user_item == "scissors" and computer_item == "paper") or
            (user_item == "paper" and computer_item == "rock")
        ):
            return "win"
        else:
            return "loss"  # Sinon, le joueur perd

    # Cette méthode joue une manche complète du jeu
    def play(self):
        """Joue une manche du jeu et affiche le résultat."""
        user_item = self.get_user_item()  # On récupère le choix du joueur
        computer_item = self.get_computer_item()  # On récupère le choix de l'ordinateur
        result = self.get_game_result(user_item, computer_item)  # On détermine le résultat du jeu

        # On affiche les choix
        print(f"You selected {user_item}. The computer selected {computer_item}.")

        # On affiche le résultat selon le cas
        if result == "win":
            print("You win!")
        elif result == "loss":
            print("You lose!")
        else:
            print("It's a draw!")

        return result  # On retourne le résultat (win/draw/loss)