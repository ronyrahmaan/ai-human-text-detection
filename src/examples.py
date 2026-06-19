"""Two sample passages so users can try the app with one click.

Both are drawn from the training dataset and are confidently classified.
"""

EXAMPLE_HUMAN = (
    "Yes, i agree with the statement \"successful people always try new things "
    "and take risks, because they will get experience, money and confidence. "
    "Successful people will gain some experience from past work, so it will help "
    "in their new things to get success. They will easily succeed in that work. "
    "They feel bore doing same work, to break the monotony they will try new "
    "things. They will take risk. With the success of past one they will get "
    "self-confident. Confidence will clear the way to the success. Successful "
    "people have also get money from past success. With that money they will do "
    "new things, these new things will also develop their knowledge. Finally, "
    "from the above points i can say the successful persons will try new things."
)

EXAMPLE_AI = (
    "For the AON model we use the code base provided by the authors and we "
    "maintain the hyperparameters described in the paper. For the paragraph "
    "encoder of the BAON models, we follow the same scheme of the AON model, but "
    "for its sentence encoder we use the hyperparameters of the BERT setting. We "
    "use the pretrained BERT uncased base model with 12 layers for the BAON and "
    "BTSORT models, and we finetune the BERT model in both cases. Hence, we "
    "replace the Adadelta optimizer with the BertAdam optimizer for the BAON "
    "model. The LSTMs in the LTSort model use an RNN size of 512 and the same "
    "vocabularies as the AON model. LTSort is trained using stochastic gradient "
    "descent with dropout of 0.2, a learning rate of 1.0, and a learning decay "
    "rate of 0.5. For all experiments we use a maximum sequence length of 105 "
    "tokens."
)
