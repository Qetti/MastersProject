# სარეკომენდაციო სისტემა პროფესიების შერჩევისთვის

ეს პროექტი წარმოადგენს სარეკომენდაციო სისტემას, რომელიც ეხმარება მომხმარებლებს პროფესიების შერჩევაში მათი პასუხების საფუძველზე.

## ინსტალაცია

1. დაკლონეთ რეპოზიტორია:

    ```
    git clone https://github.com/Qetti/MastersProject.git
    cd MastersProject
    ```
    
2. დააინსტალირეთ საჭირო პაკეტები:

    ```
    pip install -r requirements.txt
    ```

## გამოყენება

1. გაუშვით Flask აპლიკაცია:

    ```
    python app.py
    ```

2. გახსენით ვებ ბრაუზერი და გადადით მისამართზე `http://localhost:5000`

3. შეავსეთ კითხვარი და მიიღეთ პროფესიების რეკომენდაციები

## პროექტის სტრუქტურა

- `app.py`: Flask აპლიკაციის მთავარი ფაილი
- `models/`: დირექტორია, სადაც ინახება წინასწარ გაწვრთნილი მოდელები
- `templates/`: HTML შაბლონები
- `requirements.txt`: პროექტის დამოკიდებულებები

