# rock-paper-scissors.py

# On importe la classe Game depuis le fichier game.py
from game import Game

# Cette fonction affiche un petit menu et retourne le choix de l'utilisateur
def get_user_menu_choice():
    """Affiche le menu et retourne le choix de l'utilisateur."""
    print("\nMenu:")
    print("(p) Play a new game")
    print("(s) Show scores")
    print("(q) Quit")
    
    choice = input("Enter your choice: ").lower()  # On récupère la réponse en minuscule
    if choice in ['p', 's', 'q']:  # On valide la réponse
        return choice
    else:
        print("Invalid choice.")
        return None

# Cette fonction affiche un résumé des résultats
def print_results(results):
    """Affiche les résultats des parties jouées."""
    print("\nGame Results:")
    print(f"You won {results['win']} times")
    print(f"You lost {results['loss']} times")
    print(f"You drew {results['draw']} times")
    print("Thanks for playing! 👋")

# Fonction principale qui contrôle tout le programme
def main():
    """Boucle principale du jeu."""
    results = {"win": 0, "loss": 0, "draw": 0}  # On initialise les scores

    while True:  # On répète jusqu'à ce que l'utilisateur quitte
        user_choice = get_user_menu_choice()  # On récupère le choix du menu

        if user_choice == 'p':  # Jouer une partie
            game = Game()  # Crée un objet Game
            result = game.play()  # Lance une partie et récupère le résultat
            results[result] += 1  # On met à jour les scores

        elif user_choice == 's':  # Afficher les scores
            print_results(results)

        elif user_choice == 'q':  # Quitter le jeu
            print_results(results)
            break  # On sort de la boucle

# Ce bloc s'exécute uniquement si on lance ce fichier directement
if __name__ == "__main__":
    main()