import asyncio
from quizz_generator import generate_quizz
from clean_text import list_abstract_fr_clean


def transform(input_list):
    """
    Transforme la sortie brute du modèle en format structuré pour un quiz.
    """
    new_list = []
    for item in input_list:
        for key in item:
            if 'question1' in key or 'question2' in key or 'question3' in key:
                question_dict = {}
                question_num = key[-1]  # Récupérer le numéro de la question
                
                question_dict['question'] = item[key]
                question_dict['A'] = item.get(f'A_{question_num}', '')
                question_dict['B'] = item.get(f'B_{question_num}', '')
                question_dict['C'] = item.get(f'C_{question_num}', '')
                question_dict['D'] = item.get(f'D_{question_num}', '')
                question_dict['reponse'] = item.get(f'reponse{question_num}', '')

                new_list.append(question_dict)

    return new_list





async def txt_to_quizz(content):
    """
    Génère un quiz à partir d'un texte donné.
    """
    quizz = await generate_quizz(content)
    
    if quizz:
        transformed_quizz = transform(quizz)
        return transformed_quizz

    return ''

# Exécution principale
if __name__ == "__main__":
    # On utilise asyncio pour exécuter l'appel au modèle LLM
    quiz_result = asyncio.run(txt_to_quizz(list_abstract_fr_clean[:5]))  # On teste avec 5 abstracts
    print(quiz_result)