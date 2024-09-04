from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import OrderedDict

# Load pre-trained transformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Description to match
description = "When reporting to the client his quarterly fees, the document did not contain some specific fees that he has paid due to technical issues. The client has therefore asked the Bank to be reimbursed the related amounts (CHF 5'000)."

# Keywords to match
keywords = OrderedDict({
    "Operations": "process failure, human error, incorrect order execution, execution issue",
    # "Trading": "market risk, trading errors, securities, trade execution, order mistake",
    # "Client relationship management": "client service, customer satisfaction, fee dispute, communication issue, client reimbursement",
    "Technology": "technical issues, system failure, software problems, IT errors, reporting system, automated reporting error, data processing issue, missing information in report",
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
sorted_indices = np.argsort(-similarity[0].cpu().numpy())

# Output the result
for index in sorted_indices:
    category = list(keywords.keys())[index]
    sim = similarity[0, index].item()
    print(f"{category} - Similarity: {sim}")