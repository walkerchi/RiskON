import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, util
sns.set_style("darkgrid")

np.random.rand(1)

# Load pre-trained transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')



# Description to match
description = "When reporting to the client his quarterly fees, the document did not contain some specific fees that he has paid due to technical issues. The client has therefore asked the Bank to be reimbursed the related amounts (CHF 5'000)."

# Keywords to match
keywords = OrderedDict({
    "Operations": "process failure, human error, incorrect order execution, execution issue",
    # "Trading": "market risk, trading errors, securities, trade execution, order mistake",
    # "Client relationship management": "client service, customer satisfaction, fee dispute, communication issue, client reimbursement",
    "Technology": "not tehcnical issue, system failure, software problems, IT errors, reporting system, automated reporting error, data processing issue, missing information in report",
    #"Finance and treasury": "financial management, capital structure, liquidity, treasury, accounting, client fees, reimbursement, financial reporting"
})

# Combine description and keywords for embedding
sentences = [description] + list(keywords.values())

# Generate embeddings for all sentences
embeddings = model.encode(sentences)

# Separate description embedding and keyword embeddings
description_embedding = embeddings[0]
keyword_embeddings = embeddings[1:]

# Compute cosine similarity between description and each set of keywords
similarity = util.cos_sim(description_embedding, keyword_embeddings)


# Sort the results based on similarity
sorted_indices = similarity[0].argsort(descending=True)

# Output the result
for index in sorted_indices:
    category = list(keywords.keys())[index]
    sim = similarity[0, index].item()
    print(f"{category} - Similarity: {sim}")

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot()

ax.set_xticks([])
ax.set_yticks([])
all_embeddings = model.encode([
    description,
    "Operations",
    "Trading",
    "Client relationship management",
    "Technology",
    "Finance and treasury",
    "The Banker has received an unclear order instruction from his client and therefore has proceeded to a wrong order execution in the system regarding the purchase of shares in an investment fund. As a consequence, the Bank has taken in its books the position and repurchased the correct security for the client. When seeling the position retaken by the Bank, it showed that the Bank has made a loss of CHF 1'000 (market price decrease) reprsenting the final impact of this incident.",
    "The trader has incorrectly input in the system an order instruction received by a client which had as a consquence that he has not bought enough quantity of a security. Once he discovered the error, the trader has repurchased the missing securities, however the price had slightly increased. As a consequence, the Bank has compensated the client for the extra price paid (CHF 2'000).",
    "When cancelling a credit card of a client by the banker, the system has inappropriately debited the client of the fees related to the closure of a mandate. However, these fees should not have been debited to this client. As a consequence, the Bank has reimbursed the client for these incorrect fees charged (CHF 200).",
    "At the initial codification in the system of a specific client, an error was made which led the client paying extra undue taxes. The Bank has reimbursed the client for these undue taxes paid (CHF 30'000).",
    "A fund manager has inappropriately managed a compartment of a fund as he did not respect specific constraints which led to a loss that needs to be absorbed by the Bank (CHF 7'000)."

])
all_embeddings = TSNE().fit_transform(all_embeddings)
all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, ord=2,axis=-1, keepdims=True)
embeddings = all_embeddings[[0,1,4]]
description_embedding = embeddings[0:1]
keyword_embeddings = embeddings[1:]

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

ax.scatter(description_embedding[:,0],description_embedding[:,1],
           color="orange",
           s = 250,
           marker="1",
           linewidths=8,
           label="description")
ax.scatter(keyword_embeddings[:, 0], keyword_embeddings[:, 1],
           color="dodgerblue",
           s = 250,
           marker="2",
           linewidths=8,
           label="choices")
ax.text(description_embedding[0,0],description_embedding[0,1],
        "...technical issues...")
for i in range(keyword_embeddings.shape[0]):
    ax.text(keyword_embeddings[i, 0], keyword_embeddings[i, 1],
            list(keywords.keys())[i])
where_label = list(keywords.keys()).index("Operations")
ax.plot([description_embedding[0,0], keyword_embeddings[sorted_indices[0], 0]],
        [description_embedding[0,1], keyword_embeddings[sorted_indices[0], 1]],
        linestyle="--",
        alpha=0.5,
        color="green",label="Prediction")
ax.plot([description_embedding[0,0], keyword_embeddings[where_label, 0]],
        [description_embedding[0,1], keyword_embeddings[where_label, 1]],
        linestyle="-.",
        alpha=0.5,
        color="grey",label="Label")
# for i in range(embeddings.shape[0]):
#     ax.plot([0.0, embeddings[i, 0]],
#             [0.0, embeddings[i, 1]],
#             linestyle=":",
#             color="black",
#             alpha=0.5)

ax.legend()
os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/task2_example.png")