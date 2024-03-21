import tensorflow_datasets as tfds

# Download the plain text default config
imdb_plaintext, info_plaintext = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# Download the subword encoded pretokenized dataset
imdb_subwords, info_subwords = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

# Print description of features
info_plaintext.features

"""
FeaturesDict({
    'label': ClassLabel(shape=(), dtype=int64, num_classes=2),
    'text': Text(shape=(), dtype=string),
})
"""

# Take 2 training examples and print the text feature
for example in imdb_plaintext['train'].take(2):
  print(example[0].numpy())

# Print description of features
info_subwords.features

# subwords8k dataset is already tokenized
# text features includes an encoder field
"""
FeaturesDict({
    'label': ClassLabel(shape=(), dtype=int64, num_classes=2),
    'text': Text(shape=(None,), dtype=int64, encoder=<SubwordTextEncoder vocab_size=8185>),
})
"""

# Take 2 training examples and print its contents
for example in imdb_subwords['train'].take(2):
  print(example)

"""
(<tf.Tensor: shape=(163,), dtype=int64, numpy=
array([  62,   18,   41,  604,  927,   65,    3,  644, 7968,   21,   35,
       5096,   36,   11,   43, 2948, 5240,  102,   50,  681, 7862, 1244,
          3, 3266,   29,  122,  640,    2,   26,   14,  279,  438,   35,
         79,  349,  384,   11, 1991,    3,  492,   79,  122,  188,  117,
         33, 4047, 4531,   14,   65, 7968,    8, 1819, 3947,    3,   62,
         27,    9,   41,  577, 5044, 2629, 2552, 7193, 7961, 3642,    3,
         19,  107, 3903,  225,   85,  198,   72,    1, 1512,  738, 2347,
        102, 6245,    8,   85,  308,   79, 6936, 7961,   23, 4981, 8044,
          3, 6429, 7961, 1141, 1335, 1848, 4848,   55, 3601, 4217, 8050,
          2,    5,   59, 3831, 1484, 8040, 7974,  174, 5773,   22, 5240,
        102,   18,  247,   26,    4, 3903, 1612, 3902,  291,   11,    4,
         27,   13,   18, 4092, 4008, 7961,    6,  119,  213, 2774,    3,
         12,  258, 2306,   13,   91,   29,  171,   52,  229,    2, 1245,
       5790,  995, 7968,    8,   52, 2948, 5240, 8039, 7968,    8,   74,
       1249,    3,   12,  117, 2438, 1369,  192,   39, 7975])>, <tf.Tensor: shape=(), dtype=int64, numpy=0>)
(<tf.Tensor: shape=(142,), dtype=int64, numpy=
array([  12,   31,   93,  867,    7, 1256, 6585, 7961,  421,  365,    2,
         26,   14,    9,  988, 1089,    7,    4, 6728,    6,  276, 5760,
       2587,    2,   81, 6118, 8029,    2,  139, 1892, 7961,    5, 5402,
        246,   25,    1, 1771,  350,    5,  369,   56, 5397,  102,    4,
       2547,    3, 4001,   25,   14, 7822,  209,   12, 3531, 6585, 7961,
         99,    1,   32,   18, 4762,    3,   19,  184, 3223,   18, 5855,
       1045,    3, 4232, 3337,   64, 1347,    5, 1190,    3, 4459,    8,
        614,    7, 3129,    2,   26,   22,   84, 7020,    6,   71,   18,
       4924, 1160,  161,   50, 2265,    3,   12, 3983,    2,   12,  264,
         31, 2545,  261,    6,    1,   66,    2,   26,  131,  393,    1,
       5846,    6,   15,    5,  473,   56,  614,    7, 1470,    6,  116,
        285, 4755, 2088, 7961,  273,  119,  213, 3414, 7961,   23,  332,
       1019,    3,   12, 7667,  505,   14,   32,   44,  208, 7975])>, <tf.Tensor: shape=(), dtype=int64, numpy=0>)  
"""
