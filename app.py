import faiss
import pickle
import numpy as np
from flask import (
    Flask,
    render_template,
    request
)
from configs.config import (
    section1_questions,
    section2_questions,
    section1_label_mapping,
    section1_reverse_mapping
)

app = Flask(__name__)


def get_faiss_recommendations(input_features, target_labels, target_encoder, num_recommendations=3):
    """
    FAISS რეკომენდერი.
    ეს ფუნქცია იყენებს FAISS ინდექსს რეკომენდაციების მისაღებად.
    ფუნქცია აბრუნებს მსგავსი პროფესიების სიას მათი ალბათობის ქულებთან ერთად.
    """
    input_features = np.array(input_features).astype(np.float32).reshape(1, -1)
    input_features = np.ascontiguousarray(input_features)
    faiss.normalize_L2(input_features)
    k_neighbors = num_recommendations * 10

    faiss_index = faiss.read_index('models/faiss_index.index')

    distances, indices = faiss_index.search(input_features, k_neighbors)
    similarities = distances.flatten()
    indices = indices.flatten()

    sorted_indices = np.argsort(similarities)[::-1]
    unique_professions = []
    top_professions = []

    for idx in sorted_indices:
        profession = target_encoder.inverse_transform([target_labels[indices[idx]]])[0]
        if profession not in unique_professions and similarities[idx] > 0:
            unique_professions.append(profession)
            top_professions.append((profession, similarities[idx]))
            if len(top_professions) == num_recommendations:
                break

    return top_professions


def get_rf_recommendations(input_features, target_encoder):
    """
    Random Forest რეკომენდერი.
    ფუნქცია აბრუნებს სავარაუდო პროფესიების სიას მათ ალბათობებთან ერთად.
    """
    input_features = np.array(input_features).astype(np.float32).reshape(1, -1)
    with open('models/random_forest_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)

    top_professions = get_top_professions(rf_model, input_features, target_encoder)

    return top_professions


def get_lr_recommendations(input_features, target_encoder):
    """
    მულტინომიალური ლოჯისტიკური რეგრესიის რეკომენდერი.
    ეს ფუნქცია აბრუნებს პროგნოზირებული პროფესიების სიას მათი ალბათობებით.
    """
    input_features = np.array(input_features).astype(np.float32).reshape(1, -1)

    with open('models/logistic_regression_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)

    top_professions = get_top_professions(lr_model, input_features, target_encoder)

    return top_professions


def get_top_professions(model, input_features, target_encoder, num_recommendations=3):
    """
    გადაცემული მოდელისთვის აბრუნებს TOP 3 პროფესიას თავისი ალბათობის ქულით
    """
    predictions = model.predict_proba(input_features)[0]
    top_indices = np.argsort(predictions)[::-1][:num_recommendations]
    top_professions = [(target_encoder.inverse_transform([index])[0], predictions[index]) for index in top_indices
                       if predictions[index] > 0]
    top_professions = top_professions[:num_recommendations]

    return top_professions


@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)


@app.route('/')
def index1():
    """
    ეს ფუნქცია არენდერებს მთავარ გვერდს და გადასცემს შაბლონს (დეფოლტად არჩეულ პასუხებს)
    """
    default_encoded_input = [4, 4, 4, 4, 4, 3, 1, 0, 2, 0, 1, 0, 2, 1, 0, 3, 3, 3, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 4, 0, 0, 1, 3, 4, 0, 0, 1, 2, 3, 0, 0, 0, 2, 0, 0, 3, 3, 5, 5, 5, 5, 1, 1, 1, 1, 2, 5,
                             1, 3, 3, 3, 4, 3, 2, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 5, 5, 5, 1, 1, 1]

    default_input_list = []
    for i, val in enumerate(default_encoded_input):
        if i < 50:
            decoded_value = section1_reverse_mapping[val]
            default_input_list.append(decoded_value)
        else:
            default_input_list.append(str(val))

    return render_template('index.html', section1_questions=section1_questions, section2_questions=section2_questions,
                           default_input_list=default_input_list)


@app.route('/submit', methods=['POST'])
def submit():
    """
    როდესაც მომხმარებელი დააჭერს შენახვის ღილაკს, იწყებს ეს ფუნქცია მუშაობას
    ამუშავებს მომხმარებლის პასუხებს, იღებს რეკომენდაციებს FAISS და Random Forest მოდელებიდან,
    არენდერებს შედეგების გვერდს და გვიბრუნებს პასუხებს.
    """
    section1_answers = [request.form.get(f'q{i:02}') for i in range(1, 51)]
    section2_answers = [request.form.get(f'q{i:02}') for i in range(51, 91)]

    input_list = section1_answers + section2_answers
    input_list = [section1_label_mapping.get(answer, answer) for answer in input_list]
    input_list = [val if val is not None else '' for val in input_list]

    encoded_input = []
    for i, value in enumerate(input_list):
        if i < 50:
            encoded_value = value
        else:
            encoded_value = int(value)
        encoded_input.append(encoded_value)

    print("Encoded Input: ", encoded_input)

    with open('models/target_labels.pkl', 'rb') as file:
        target_labels = pickle.load(file)

    with open('models/target_encoder.pkl', 'rb') as file:
        target_encoder = pickle.load(file)

    faiss_recommendations = get_faiss_recommendations(encoded_input, target_labels, target_encoder)

    rf_recommendations = get_rf_recommendations(encoded_input, target_encoder)

    lr_recommendations = get_lr_recommendations(encoded_input, target_encoder)

    profession_map = {
        'a': 'პროგრამისტი',
        'b': 'ექიმი',
        'g': 'ფინანსური ანალიტიკოსი',
        'd': 'მასწავლებელი',
        'e': 'არქიტექტორი',
        'v': 'იურისტი',
        'z': 'ფსიქოლოგი',
        't': 'ჟურნალისტი',
        'i': 'ინჟინერი',
        'k': 'დიზაინერი'
    }

    faiss_recommendations_mapped = [(profession_map[profession], f"{score:.2f}") for profession, score in
                                    faiss_recommendations]
    rf_recommendations_mapped = [(profession_map[profession], f"{score:.2f}") for profession, score in
                                 rf_recommendations]
    lr_recommendations_mapped = [(profession_map[profession], f"{score:.2f}") for profession, score in
                                 lr_recommendations]

    return render_template('results.html',
                           faiss_recommendations=faiss_recommendations_mapped,
                           rf_recommendations=rf_recommendations_mapped,
                           lr_recommendations=lr_recommendations_mapped)


if __name__ == '__main__':
    app.run(debug=True)
