
const data = {
            //==== ML Concepts ====

  "ml-concepts-1": {
    id: "ml-concepts-1",
    title: "What is Machine Learning?",
    image: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60",
    sections: [
      { type: "title", content: "What is Machine Learning?" },
      {
        type: "text",
        content:
          "Machine Learning is a branch of AI that enables systems to learn from data. \\" +
          "Instead of being explicitly programmed, ML algorithms identify patterns. \\" +
          "They improve their performance with experience over time. \\" +
          "ML powers recommendations, predictions, and automated decisions. \\" +
          "It bridges the gap between data and actionable insights.",
      },
      { type: "subtitle", content: "Core Idea" },
      {
        type: "text",
        content:
          "ML algorithms find mathematical patterns in datasets. \\" +
          "They generalize from examples to make predictions on new data. \\" +
          "The learning process adjusts internal parameters based on feedback. \\" +
          "Models can be supervised, unsupervised, or reinforced. \\" +
          "The goal is to automate decision-making where programming rules fail.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Email spam filters use ML to classify incoming messages. \\" +
          "The system learns from users marking emails as spam or not. \\" +
          "Patterns in sender addresses, content, and subject lines are analyzed. \\" +
          "Over time, the filter becomes more accurate at blocking unwanted emails. \\" +
          "This reduces inbox clutter without manual rule setting.",
      },
    ],
  },
  "ml-concepts-2": {
    id: "ml-concepts-2",
    title: "Supervised vs Unsupervised Learning",
    image: "https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60",
    sections: [
      { type: "title", content: "Supervised vs Unsupervised Learning" },
      {
        type: "text",
        content:
          "Supervised learning uses labeled data to train prediction models. \\" +
          "The algorithm learns from input-output pairs provided during training. \\" +
          "Unsupervised learning finds hidden patterns in unlabeled data. \\" +
          "It groups similar items or reduces data dimensionality. \\" +
          "Both approaches solve different types of real-world problems.",
      },
      { type: "subtitle", content: "Key Differences" },
      {
        type: "text",
        content:
          "Supervised: Needs labeled data, predicts known outcomes. \\" +
          "Unsupervised: Works with unlabeled data, discovers unknown patterns. \\" +
          "Supervised examples: Classification and regression tasks. \\" +
          "Unsupervised examples: Clustering and association tasks. \\" +
          "Semi-supervised learning combines both approaches.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Amazon uses supervised learning to recommend products. \\" +
          "Customer purchase history provides labeled training data. \\" +
          "Unsupervised learning groups similar products for discovery. \\" +
          "Combined approaches create personalized shopping experiences. \\" +
          "This drives higher engagement and conversion rates.",
      },
    ],
  },
  "ml-concepts-3": {
    id: "ml-concepts-3",
    title: "Regression & Classification",
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60",
    sections: [
      { type: "title", content: "Regression & Classification" },
      {
        type: "text",
        content:
          "Regression predicts continuous numerical values like prices or temperatures. \\" +
          "Classification predicts discrete categories like spam/not-spam. \\" +
          "Both are supervised learning techniques with different outputs. \\" +
          "Regression uses algorithms like linear regression and decision trees. \\" +
          "Classification employs logistic regression, SVM, and neural networks.",
      },
      { type: "subtitle", content: "When to Use Each" },
      {
        type: "text",
        content:
          "Use regression for quantity predictions: stock prices, sales forecasts. \\" +
          "Use classification for category predictions: disease diagnosis, sentiment. \\" +
          "Regression minimizes distance between predicted and actual values. \\" +
          "Classification maximizes accuracy of category assignments. \\" +
          "Many real problems combine both approaches.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Uber uses regression to predict ride prices and arrival times. \\" +
          "The model considers traffic, demand, distance, and historical patterns. \\" +
          "Classification determines ride categories like pool, premier, or XL. \\" +
          "Both systems work together to optimize user experience. \\" +
          "This creates efficient, predictable transportation services.",
      },
    ],
  },
  "ml-concepts-4": {
    id: "ml-concepts-4",
    title: "Clustering & Dimensionality Reduction",
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60",
    sections: [
      { type: "title", content: "Clustering & Dimensionality Reduction" },
      {
        type: "text",
        content:
          "Clustering groups similar data points without predefined labels. \\" +
          "K-means and hierarchical clustering are popular algorithms. \\" +
          "Dimensionality reduction simplifies data while preserving patterns. \\" +
          "PCA and t-SNE reduce features for visualization and efficiency. \\" +
          "Both are unsupervised techniques for data exploration.",
      },
      { type: "subtitle", content: "Applications" },
      {
        type: "text",
        content:
          "Clustering: Customer segmentation, document grouping, anomaly detection. \\" +
          "Dimensionality reduction: Data visualization, noise reduction, feature extraction. \\" +
          "Clustering finds natural groupings in complex datasets. \\" +
          "Dimensionality reduction prevents overfitting in high-dimensional data. \\" +
          "Together they reveal hidden data structure.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Spotify uses clustering to create personalized playlists. \\" +
          "Songs with similar audio features are grouped together. \\" +
          "Dimensionality reduction helps visualize music taste patterns. \\" +
          "Users discover new music through algorithmic recommendations. \\" +
          "This keeps listeners engaged with fresh, relevant content.",
      },
    ],
  },
  "ml-concepts-5": {
    id: "ml-concepts-5",
    title: "Model Evaluation Metrics",
    image: "https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60",
    sections: [
      { type: "title", content: "Model Evaluation Metrics" },
      {
        type: "text",
        content:
          "Evaluation metrics quantify how well ML models perform. \\" +
          "Different metrics suit different problem types and business goals. \\" +
          "Accuracy measures overall correctness but can be misleading. \\" +
          "Precision and recall provide nuanced classification insights. \\" +
          "The right metric aligns with real-world impact.",
      },
      { type: "subtitle", content: "Common Metrics" },
      {
        type: "text",
        content:
          "Accuracy: Proportion of correct predictions overall. \\" +
          "Precision: How many selected items are relevant. \\" +
          "Recall: How many relevant items are selected. \\" +
          "F1-Score: Harmonic mean of precision and recall. \\" +
          "ROC-AUC: Overall classification performance across thresholds.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Medical diagnosis systems prioritize recall over accuracy. \\" +
          "Missing a disease (false negative) is more costly than false alarm. \\" +
          "Cancer screening algorithms maximize detection of true positives. \\" +
          "Evaluation metrics guide model selection and improvement. \\" +
          "This saves lives through early, accurate disease detection.",
      },
    ],
  },
  "ml-concepts-6": {
    id: "ml-concepts-6",
    title: "Bias-Variance Tradeoff",
    image: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60",
    sections: [
      { type: "title", content: "Bias-Variance Tradeoff" },
      {
        type: "text",
        content:
          "Bias is error from overly simplistic model assumptions. \\" +
          "Variance is error from excessive sensitivity to training data. \\" +
          "High bias causes underfitting; high variance causes overfitting. \\" +
          "The tradeoff balances model complexity and generalization. \\" +
          "Optimal models minimize total error (bias² + variance + irreducible error).",
      },
      { type: "subtitle", content: "Managing the Tradeoff" },
      {
        type: "text",
        content:
          "Increase model complexity to reduce bias. \\" +
          "Add regularization to reduce variance. \\" +
          "Use cross-validation to estimate true performance. \\" +
          "Collect more data to reduce variance impact. \\" +
          "Choose algorithms based on data characteristics.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Credit scoring models balance bias-variance carefully. \\" +
          "Too simple (high bias) misses important risk factors. \\" +
          "Too complex (high variance) overfits to historical patterns. \\" +
          "Banks optimize for fair, generalizable lending decisions. \\" +
          "This maximizes profitability while minimizing default risk.",
      },
    ],
  },
  "ml-concepts-7": {
    id: "ml-concepts-7",
    title: "Cross-Validation Techniques",
    image: "https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60",
    sections: [
      { type: "title", content: "Cross-Validation Techniques" },
      {
        type: "text",
        content:
          "Cross-validation estimates model performance on unseen data. \\" +
          "It partitions data into training and validation sets multiple times. \\" +
          "K-fold cross-validation is the most common approach. \\" +
          "Stratified cross-validation preserves class distribution. \\" +
          "Cross-validation prevents overfitting and provides robust performance estimates.",
      },
      { type: "subtitle", content: "Types of Cross-Validation" },
      {
        type: "text",
        content:
          "K-Fold: Data split into K equal parts, each used as validation once. \\" +
          "Leave-One-Out: Each sample serves as validation individually. \\" +
          "Stratified: Maintains class proportions in each fold. \\" +
          "Time Series: Respects temporal ordering for sequential data. \\" +
          "Nested: Tunes hyperparameters without data leakage.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Pharmaceutical companies use cross-validation in drug discovery. \\" +
          "Molecular activity predictions must generalize to new compounds. \\" +
          "Cross-validation ensures models work beyond training molecules. \\" +
          "This reduces failed clinical trials and accelerates development. \\" +
          "Life-saving medications reach patients faster through robust validation.",
      },
    ],
  },
  "ml-concepts-8": {
    id: "ml-concepts-8",
    title: "Feature Engineering Basics",
    image: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60",
    sections: [
      { type: "title", content: "Feature Engineering Basics" },
      {
        type: "text",
        content:
          "Feature engineering creates informative input variables from raw data. \\" +
          "Good features dramatically improve model performance. \\" +
          "It involves transformation, creation, and selection of variables. \\" +
          "Domain knowledge guides effective feature engineering. \\" +
          "Well-engineered features often outperform complex algorithms.",
      },
      { type: "subtitle", content: "Common Techniques" },
      {
        type: "text",
        content:
          "One-hot encoding for categorical variables. \\" +
          "Scaling and normalization for numerical features. \\" +
          "Polynomial features for capturing interactions. \\" +
          "Binning continuous variables into categories. \\" +
          "Creating time-based features from timestamps.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Airbnb uses feature engineering for price prediction. \\" +
          "Location features include distance to landmarks and neighborhood scores. \\" +
          "Temporal features capture seasonality and local events. \\" +
          "Property features combine amenities, photos, and host history. \\" +
          "Accurate pricing maximizes bookings and host revenue.",
      },
    ],
  },
  "ml-concepts-9": {
    id: "ml-concepts-9",
    title: "Ensemble Methods",
    image: "https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60",
    sections: [
      { type: "title", content: "Ensemble Methods" },
      {
        type: "text",
        content:
          "Ensemble methods combine multiple models to improve predictions. \\" +
          "They reduce variance, bias, and overfitting through aggregation. \\" +
          "Random forests average many decision trees. \\" +
          "Gradient boosting sequentially corrects previous errors. \\" +
          "Ensembles often win machine learning competitions.",
      },
      { type: "subtitle", content: "Popular Ensemble Techniques" },
      {
        type: "text",
        content:
          "Bagging: Parallel training of same algorithm on different subsets. \\" +
          "Boosting: Sequential training focusing on hard examples. \\" +
          "Stacking: Combining predictions using a meta-model. \\" +
          "Voting: Majority or weighted average of predictions. \\" +
          "Each technique addresses different weaknesses.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Netflix recommendation system uses ensemble methods. \\" +
          "Multiple algorithms predict user ratings for content. \\" +
          "Combined predictions improve accuracy beyond any single model. \\" +
          "The ensemble adapts to diverse viewing patterns. \\" +
          "This creates highly personalized entertainment experiences.",
      },
    ],
  },
  "ml-concepts-10": {
    id: "ml-concepts-10",
    title: "ML in Everyday Life",
    image: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60",
    sections: [
      { type: "title", content: "ML in Everyday Life" },
      {
        type: "text",
        content:
          "Machine learning quietly powers daily experiences and services. \\" +
          "From search engines to social media, ML is everywhere. \\" +
          "It personalizes content, optimizes routes, and detects fraud. \\" +
          "ML improves efficiency, safety, and convenience across industries. \\" +
          "Understanding ML helps navigate the increasingly AI-driven world.",
      },
      { type: "subtitle", content: "Ubiquitous Applications" },
      {
        type: "text",
        content:
          "Search engines rank results using ML algorithms. \\" +
          "Social media curates feeds based on engagement patterns. \\" +
          "Navigation apps predict traffic and optimize routes. \\" +
          "Smart assistants understand and respond to voice commands. \\" +
          "Streaming services recommend content you will enjoy.",
      },
      { type: "image", content: "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60" },
      { type: "subtitle", content: "Real-Time Example" },
      {
        type: "text",
        content:
          "Google Maps uses ML for real-time traffic predictions. \\" +
          "Historical data, current speeds, and events inform ETAs. \\" +
          "The system learns traffic patterns for different days and times. \\" +
          "Accurate predictions save millions of hours in commute time. \\" +
          "This demonstrates ML solving everyday transportation challenges.",
      },
    ],
  },


  //==== Deep Learnig ====
  'deep-learning-1': {
  id: 'deep-learning-1',
  title: 'What Makes Learning "Deep"?',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'What Makes Learning "Deep"?' },
    { type: 'text', content:
      'Deep learning uses neural networks with many hidden layers. \
These layers enable hierarchical feature learning from raw data. \
Deep networks automatically discover representations needed for tasks. \
They eliminate manual feature engineering for complex problems. \
Depth allows modeling of highly non-linear relationships.' 
    },
    { type: 'subtitle', content: 'Key Characteristics' },
    { type: 'text', content:
      'Multiple processing layers for feature transformation. \
Automatic feature extraction from raw inputs. \
Ability to learn hierarchical representations. \
Scalability with large datasets and computational resources. \
State-of-the-art performance in perception tasks.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Translate uses deep learning for accurate translations. \
Multiple layers understand grammar, syntax, and semantics. \
The system learns language patterns from millions of documents. \
Deep networks capture nuances missed by earlier approaches. \
This enables near-human translation across 100+ languages.' 
    }
  ]
},
'deep-learning-2': {
  id: 'deep-learning-2',
  title: 'Convolutional Neural Networks (CNNs)',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Convolutional Neural Networks (CNNs)' },
    { type: 'text', content:
      'CNNs are specialized for processing grid-like data such as images. \
They use convolutional filters to detect spatial patterns. \
Pooling layers reduce dimensionality while preserving features. \
CNNs achieve state-of-the-art results in computer vision tasks. \
They revolutionized image recognition and processing.' 
    },
    { type: 'subtitle', content: 'CNN Architecture Components' },
    { type: 'text', content:
      'Convolutional layers detect local patterns. \
Activation layers introduce non-linearity (ReLU). \
Pooling layers downsample feature maps. \
Fully connected layers make final predictions. \
Batch normalization stabilizes training.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Facebook uses CNNs for automatic photo tagging. \
The system recognizes faces, objects, and scenes in images. \
Multiple convolutional layers extract features hierarchically. \
This enables instant friend suggestions and content organization. \
CNNs power the social media visual experience.' 
    }
  ]
},
'deep-learning-3': {
  id: 'deep-learning-3',
  title: 'Recurrent Neural Networks (RNNs)',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Recurrent Neural Networks (RNNs)' },
    { type: 'text', content:
      'RNNs process sequential data by maintaining internal memory. \
They handle inputs of varying lengths through recurrent connections. \
RNNs excel at time-series analysis and natural language tasks. \
They capture temporal dependencies and contextual information. \
Variants like LSTMs address long-term dependency problems.' 
    },
    { type: 'subtitle', content: 'RNN Applications' },
    { type: 'text', content:
      'Time series forecasting and anomaly detection. \
Speech recognition and audio processing. \
Natural language understanding and generation. \
Video analysis across time frames. \
Musical composition and sequence generation.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Apple Siri uses RNNs for voice recognition and understanding. \
The system processes audio streams as sequential data. \
RNNs maintain context throughout spoken sentences. \
This enables natural conversation with virtual assistants. \
Deep learning makes human-computer interaction seamless.' 
    }
  ]
},
'deep-learning-4': {
  id: 'deep-learning-4',
  title: 'Long Short-Term Memory (LSTMs)',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Long Short-Term Memory (LSTMs)' },
    { type: 'text', content:
      'LSTMs are specialized RNNs designed to remember long-term dependencies. \
They use gating mechanisms to control information flow. \
Input, forget, and output gates regulate memory cell updates. \
LSTMs solve the vanishing gradient problem in standard RNNs. \
They excel at tasks requiring memory over long sequences.' 
    },
    { type: 'subtitle', content: 'LSTM Advantages' },
    { type: 'text', content:
      'Remember information for many time steps. \
Learn which information to keep or discard. \
Handle sequences with long-term dependencies. \
Mitigate vanishing and exploding gradients. \
Achieve better performance on complex sequential tasks.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Keyboard uses LSTMs for predictive text and autocorrect. \
The system remembers typing patterns and context. \
LSTMs learn from millions of users typing behaviors. \
Predictions become personalized and contextually aware. \
This speeds up mobile communication through intelligent assistance.' 
    }
  ]
},
'deep-learning-5': {
  id: 'deep-learning-5',
  title: 'Autoencoders & Dimensionality Reduction',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Autoencoders & Dimensionality Reduction' },
    { type: 'text', content:
      'Autoencoders are neural networks that learn efficient data representations. \
They consist of encoder and decoder components. \
The bottleneck layer creates compressed feature representations. \
Autoencoders excel at denoising and anomaly detection. \
They provide nonlinear dimensionality reduction.' 
    },
    { type: 'subtitle', content: 'Autoencoder Types' },
    { type: 'text', content:
      'Vanilla autoencoders for basic compression. \
Denoising autoencoders for robust representations. \
Variational autoencoders for generative modeling. \
Sparse autoencoders for feature discovery. \
Contractive autoencoders for invariant representations.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Credit card companies use autoencoders for fraud detection. \
The system learns normal transaction patterns. \
Anomalies trigger fraud alerts for investigation. \
Autoencoders detect novel fraud schemes without labeled examples. \
This protects customers from unauthorized transactions.' 
    }
  ]
},
'deep-learning-6': {
  id: 'deep-learning-6',
  title: 'Generative Adversarial Networks (GANs)',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Generative Adversarial Networks (GANs)' },
    { type: 'text', content:
      'GANs consist of generator and discriminator networks in competition. \
The generator creates synthetic data resembling real examples. \
The discriminator distinguishes real from generated data. \
Training progresses until the generator produces convincing fakes. \
GANs enable realistic image, video, and audio generation.' 
    },
    { type: 'subtitle', content: 'GAN Applications' },
    { type: 'text', content:
      'Photorealistic image generation from descriptions. \
Data augmentation for limited training datasets. \
Style transfer and artistic creation. \
Super-resolution and image enhancement. \
Drug discovery through molecular generation.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'NVIDIA uses GANs for AI-generated human faces and scenes. \
The system creates photorealistic images of non-existent people. \
Entertainment industry uses this for character creation. \
GANs generate training data for other AI systems. \
This demonstrates deep learning creative capabilities.' 
    }
  ]
},
'deep-learning-7': {
  id: 'deep-learning-7',
  title: 'Transfer Learning Strategies',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Transfer Learning Strategies' },
    { type: 'text', content:
      'Transfer learning applies knowledge from one task to another. \
Pre-trained models on large datasets provide excellent starting points. \
Fine-tuning adapts these models to specific domains with less data. \
Feature extraction uses pre-trained layers as fixed feature extractors. \
Transfer learning democratizes deep learning for limited-data scenarios.' 
    },
    { type: 'subtitle', content: 'Transfer Learning Approaches' },
    { type: 'text', content:
      'Fine-tuning: Update all or some layers on new data. \
Feature extraction: Use pre-trained layers as fixed encoders. \
Domain adaptation: Adjust models for different data distributions. \
Multi-task learning: Share representations across related tasks. \
This reduces training time and data requirements.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Medical imaging startups use transfer learning for diagnosis. \
Models pre-trained on ImageNet adapt to X-rays and MRIs. \
Limited medical imaging data becomes sufficient for training. \
This accelerates AI adoption in healthcare. \
Transfer learning saves lives through accessible medical AI.' 
    }
  ]
},
'deep-learning-8': {
  id: 'deep-learning-8',
  title: 'Attention Mechanisms',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Attention Mechanisms' },
    { type: 'text', content:
      'Attention mechanisms allow models to focus on relevant input parts. \
They assign different weights to different sequence positions. \
Attention improves performance on long sequences and complex tasks. \
Self-attention captures relationships within a single sequence. \
Transformers rely entirely on attention mechanisms.' 
    },
    { type: 'subtitle', content: 'Attention Benefits' },
    { type: 'text', content:
      'Handle variable-length sequences effectively. \
Improve interpretability through attention visualization. \
Capture long-range dependencies better than RNNs. \
Enable parallel computation during training. \
Achieve state-of-the-art results in NLP tasks.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Search uses attention mechanisms for understanding queries. \
The system focuses on key terms and their relationships. \
Attention weights determine which words matter most. \
This improves search relevance and result quality. \
Attention makes search engines more intelligent and responsive.' 
    }
  ]
},
'deep-learning-9': {
  id: 'deep-learning-9',
  title: 'Transformers Architecture',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Transformers Architecture' },
    { type: 'text', content:
      'Transformers use attention mechanisms without recurrence. \
They process entire sequences in parallel for efficiency. \
Positional encoding provides sequence order information. \
Multi-head attention captures different representation subspaces. \
Transformers dominate modern natural language processing.' 
    },
    { type: 'subtitle', content: 'Transformer Components' },
    { type: 'text', content:
      'Self-attention layers for sequence modeling. \
Feed-forward networks for transformation. \
Layer normalization for training stability. \
Residual connections for gradient flow. \
Encoder-decoder structure for sequence-to-sequence tasks.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'OpenAI GPT models use transformer architecture for text generation. \
These models power chatbots, content creation, and code generation. \
Transformers understand context across thousands of words. \
They generate human-like text for various applications. \
This architecture enables today most advanced language AI.' 
    }
  ]
},
'deep-learning-10': {
  id: 'deep-learning-10',
  title: 'Deep Learning Hardware & Optimization',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Deep Learning Hardware & Optimization' },
    { type: 'text', content:
      'Deep learning demands specialized hardware for efficient computation. \
GPUs accelerate matrix operations central to neural networks. \
TPUs offer even faster training for large-scale models. \
Optimization techniques reduce memory and computation requirements. \
Hardware-software co-design enables practical deep learning deployment.' 
    },
    { type: 'subtitle', content: 'Optimization Techniques' },
    { type: 'text', content:
      'Mixed precision training reduces memory usage. \
Model pruning removes unnecessary weights. \
Quantization decreases precision for faster inference. \
Knowledge distillation transfers knowledge to smaller models. \
Architecture search automates optimal model design.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Tesla Autopilot uses optimized deep learning for real-time driving. \
Specialized hardware processes multiple camera streams simultaneously. \
Model optimization enables decisions within milliseconds. \
This ensures safe autonomous driving in complex environments. \
Hardware advances make real-world AI applications possible.' 
    }
  ]
},

    //==== Natural Language Processing ====

    'nlp-1': {
  id: 'nlp-1',
  title: 'Introduction to Natural Language Processing',
  image: 'https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Introduction to Natural Language Processing' },
    { type: 'text', content:
      'NLP enables computers to understand, interpret, and generate human language. \
It combines linguistics, computer science, and artificial intelligence. \
NLP powers chatbots, translation, sentiment analysis, and voice assistants. \
The field has evolved from rule-based systems to deep learning models. \
Modern NLP handles language complexity with remarkable accuracy.' 
    },
    { type: 'subtitle', content: 'Core NLP Tasks' },
    { type: 'text', content:
      'Text classification: Categorizing documents or sentences. \
Named entity recognition: Identifying people, places, organizations. \
Sentiment analysis: Determining emotional tone in text. \
Machine translation: Converting between languages. \
Text generation: Creating coherent language automatically.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Grammarly uses NLP to check writing for errors and improvements. \
The system analyzes grammar, style, tone, and clarity. \
Advanced models understand context and intent behind words. \
Millions use NLP-powered tools for better communication. \
This demonstrates NLP practical value in everyday writing.' 
    }
  ]
},
'nlp-2': {
  id: 'nlp-2',
  title: 'Text Preprocessing & Tokenization',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Text Preprocessing & Tokenization' },
    { type: 'text', content:
      'Text preprocessing prepares raw text for NLP models. \
Tokenization splits text into meaningful units like words or subwords. \
Cleaning removes noise such as special characters and HTML tags. \
Normalization standardizes text through lowercasing and stemming. \
Quality preprocessing significantly impacts model performance.' 
    },
    { type: 'subtitle', content: 'Preprocessing Steps' },
    { type: 'text', content:
      'Tokenization: Breaking text into words, subwords, or characters. \
Lowercasing: Converting all text to lowercase for consistency. \
Stopword removal: Eliminating common but uninformative words. \
Stemming/Lemmatization: Reducing words to root forms. \
Noise removal: Cleaning HTML, URLs, and special characters.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Search preprocesses billions of queries daily. \
Tokenization breaks queries into searchable terms. \
Stemming ensures "running" matches "run" in results. \
This preprocessing enables fast, accurate search results. \
Users find relevant information through optimized text processing.' 
    }
  ]
},
'nlp-3': {
  id: 'nlp-3',
  title: 'Word Embeddings & Vector Representations',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Word Embeddings & Vector Representations' },
    { type: 'text', content:
      'Word embeddings represent words as dense vectors in continuous space. \
Similar words have similar vector representations. \
Word2Vec, GloVe, and FastText are popular embedding methods. \
Embeddings capture semantic and syntactic relationships. \
They provide foundational representations for downstream NLP tasks.' 
    },
    { type: 'subtitle', content: 'Embedding Properties' },
    { type: 'text', content:
      'Similar words cluster together in vector space. \
Mathematical operations capture relationships (king - man + woman = queen). \
Dimensionality typically ranges from 50 to 300 dimensions. \
Pre-trained embeddings exist for many languages. \
Contextual embeddings like BERT capture polysemy.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Amazon uses word embeddings for product search and recommendations. \
"Wireless headphones" embeddings relate to "Bluetooth earbuds." \
The system understands synonyms and related products. \
This improves search relevance and cross-selling opportunities. \
Embeddings power intelligent e-commerce experiences.' 
    }
  ]
},
'nlp-4': {
  id: 'nlp-4',
  title: 'Sequence-to-Sequence Models',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Sequence-to-Sequence Models' },
    { type: 'text', content:
      'Seq2seq models transform one sequence into another sequence. \
They use encoder-decoder architecture with RNNs or transformers. \
Applications include translation, summarization, and dialogue systems. \
Attention mechanisms improve performance on long sequences. \
Seq2seq enables many practical NLP applications.' 
    },
    { type: 'subtitle', content: 'Seq2seq Applications' },
    { type: 'text', content:
      'Machine translation between different languages. \
Text summarization for long documents. \
Chatbot response generation in conversations. \
Speech recognition converting audio to text. \
Code generation from natural language descriptions.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Microsoft Translator uses seq2seq models for real-time translation. \
The encoder processes source language sentences. \
The decoder generates target language translations. \
Attention focuses on relevant parts of the input. \
This enables seamless cross-language communication.' 
    }
  ]
},
'nlp-5': {
  id: 'nlp-5',
  title: 'Transformer Models (BERT, GPT)',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Transformer Models (BERT, GPT)' },
    { type: 'text', content:
      'Transformers revolutionized NLP with attention-based architecture. \
BERT uses bidirectional context for understanding language. \
GPT generates coherent text through autoregressive modeling. \
These models achieve state-of-the-art results across NLP benchmarks. \
Pre-trained transformers enable transfer learning for various tasks.' 
    },
    { type: 'subtitle', content: 'Transformer Variants' },
    { type: 'text', content:
      'BERT: Bidirectional training for language understanding. \
GPT: Autoregressive generation for text creation. \
T5: Text-to-text framework unifying NLP tasks. \
RoBERTa: Optimized BERT with better training procedures. \
DistilBERT: Smaller, faster version of BERT.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Search uses BERT to understand search intent better. \
The model comprehends prepositions and context in queries. \
"Parking on a hill with no curb" differs from "parking on a hill." \
BERT captures these nuances for more relevant results. \
This improves search quality for complex, conversational queries.' 
    }
  ]
},
'nlp-6': {
  id: 'nlp-6',
  title: 'Sentiment Analysis & Text Classification',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Sentiment Analysis & Text Classification' },
    { type: 'text', content:
      'Sentiment analysis detects emotional tone in text (positive, negative, neutral). \
Text classification categorizes documents into predefined classes. \
Both tasks use supervised learning with labeled datasets. \
Applications include review analysis, content moderation, and spam detection. \
Modern approaches use deep learning for high accuracy.' 
    },
    { type: 'subtitle', content: 'Analysis Techniques' },
    { type: 'text', content:
      'Rule-based: Lexicon approaches using sentiment dictionaries. \
Machine learning: Traditional classifiers like SVM and random forests. \
Deep learning: RNNs, CNNs, and transformers for complex patterns. \
Aspect-based: Analyzing sentiment toward specific entities. \
Multilingual: Sentiment analysis across different languages.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Twitter uses sentiment analysis to gauge public opinion trends. \
The system analyzes millions of tweets about brands, events, and topics. \
Real-time sentiment tracking informs marketing and crisis management. \
Businesses adjust strategies based on public perception. \
This demonstrates NLP value in understanding mass communication.' 
    }
  ]
},
'nlp-7': {
  id: 'nlp-7',
  title: 'Named Entity Recognition (NER)',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Named Entity Recognition (NER)' },
    { type: 'text', content:
      'NER identifies and classifies named entities in text. \
Common entity types include persons, organizations, locations, dates. \
NER extracts structured information from unstructured text. \
Applications include information extraction, search, and knowledge graphs. \
State-of-the-art NER uses transformer-based models.' 
    },
    { type: 'subtitle', content: 'Entity Types' },
    { type: 'text', content:
      'Persons: Names of people (Barack Obama, Marie Curie). \
Organizations: Companies, institutions (Google, United Nations). \
Locations: Countries, cities, landmarks (Paris, Mount Everest). \
Dates/Times: Specific dates, periods (July 2023, next Monday). \
Numerical values: Money, percentages, quantities ($1 million, 25%).' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'LinkedIn uses NER to extract skills and experience from profiles. \
The system identifies job titles, companies, and locations automatically. \
This powers job matching and professional networking features. \
Recruiters find candidates with specific skill combinations. \
NER transforms unstructured resumes into searchable structured data.' 
    }
  ]
},
'nlp-8': {
  id: 'nlp-8',
  title: 'Text Summarization Techniques',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Text Summarization Techniques' },
    { type: 'text', content:
      'Text summarization creates concise versions of longer documents. \
Extractive summarization selects important sentences from the original. \
Abstractive summarization generates new sentences capturing key ideas. \
Applications include news aggregation, document analysis, and research. \
Modern approaches use sequence-to-sequence models with attention.' 
    },
    { type: 'subtitle', content: 'Summarization Approaches' },
    { type: 'text', content:
      'Extractive: Selects existing sentences using scoring methods. \
Abstractive: Generates new text using language models. \
Single-document: Summarizes individual articles or reports. \
Multi-document: Combines information from multiple sources. \
Query-focused: Summarizes documents relative to specific questions.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'News apps like Flipboard use summarization to create digests. \
The system extracts key points from multiple news sources. \
Users get comprehensive overviews without reading full articles. \
This saves time while keeping people informed. \
Summarization enables efficient information consumption.' 
    }
  ]
},
'nlp-9': {
  id: 'nlp-9',
  title: 'Question Answering Systems',
  image: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Question Answering Systems' },
    { type: 'text', content:
      'QA systems answer questions posed in natural language. \
They retrieve relevant information and generate precise answers. \
Open-domain QA works across general knowledge sources. \
Closed-domain QA specializes in specific domains like medicine or law. \
Modern QA uses reading comprehension and knowledge retrieval.' 
    },
    { type: 'subtitle', content: 'QA System Types' },
    { type: 'text', content:
      'Retrieval-based: Finds answers in existing knowledge bases. \
Generative: Creates answers using language models. \
Extractive: Identifies answer spans within provided context. \
Open-domain: Answers questions about anything using the web. \
Closed-domain: Specializes in specific topics or documents.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'IBM Watson uses QA systems for medical diagnosis support. \
Doctors ask questions about symptoms and test results. \
The system retrieves relevant medical literature and guidelines. \
Answers include confidence scores and supporting evidence. \
This assists healthcare professionals in complex decision-making.' 
    }
  ]
},
'nlp-10': {
  id: 'nlp-10',
  title: 'Chatbots & Dialogue Systems',
  image: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Chatbots & Dialogue Systems' },
    { type: 'text', content:
      'Chatbots simulate human conversation through text or voice interfaces. \
Rule-based chatbots follow predefined decision trees. \
AI-powered chatbots use machine learning for natural conversations. \
Dialogue systems maintain context across multiple exchanges. \
Applications include customer service, personal assistants, and entertainment.' 
    },
    { type: 'subtitle', content: 'Chatbot Architecture' },
    { type: 'text', content:
      'Natural language understanding interprets user input. \
Dialogue management maintains conversation state and context. \
Natural language generation creates appropriate responses. \
Integration with knowledge bases and APIs for information retrieval. \
Personality design for engaging user experiences.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Bank of America Erica assists customers with banking tasks. \
The chatbot handles balance inquiries, transfers, and bill payments. \
Natural conversations replace complex menu navigation. \
Erica learns from interactions to improve over time. \
This demonstrates NLP transforming customer service industries.' 
    }
  ]
},

//==== Computer Vision ====

'computer-vision-1': {
  id: 'computer-vision-1',
  title: 'Introduction to Computer Vision',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Introduction to Computer Vision' },
    { type: 'text', content:
      'Computer Vision enables machines to interpret and understand visual information. \
It processes digital images and videos to extract meaningful insights. \
CV combines image processing, pattern recognition, and machine learning. \
Applications range from facial recognition to autonomous vehicles. \
The field has advanced dramatically with deep learning techniques.' 
    },
    { type: 'subtitle', content: 'Core CV Tasks' },
    { type: 'text', content:
      'Image classification: Identifying objects in images. \
Object detection: Locating and classifying multiple objects. \
Image segmentation: Partitioning images into meaningful regions. \
Facial recognition: Identifying or verifying individuals. \
Scene understanding: Interpreting complete visual scenes.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'iPhone Face ID uses computer vision for secure authentication. \
The system projects 30,000 invisible dots to create a 3D face map. \
Neural networks verify identity within milliseconds. \
This replaces passwords with biometric security. \
Computer vision makes devices more secure and convenient.' 
    }
  ]
},
'computer-vision-2': {
  id: 'computer-vision-2',
  title: 'Image Processing Fundamentals',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Image Processing Fundamentals' },
    { type: 'text', content:
      'Image processing manipulates digital images to enhance or extract information. \
Basic operations include filtering, transformation, and enhancement. \
Color spaces (RGB, HSV, grayscale) represent images differently. \
Filters remove noise, sharpen edges, or blur images. \
These fundamentals underpin all computer vision applications.' 
    },
    { type: 'subtitle', content: 'Processing Techniques' },
    { type: 'text', content:
      'Filtering: Smoothing, sharpening, edge detection. \
Morphological operations: Dilation, erosion for shape analysis. \
Thresholding: Converting to binary images. \
Histogram equalization: Improving contrast. \
Geometric transformations: Rotation, scaling, perspective correction.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Instagram filters use image processing for creative effects. \
The app applies color adjustments, blur effects, and overlays. \
Real-time processing happens on mobile devices. \
Users transform ordinary photos into artistic creations. \
This demonstrates image processing consumer applications.' 
    }
  ]
},
'computer-vision-3': {
  id: 'computer-vision-3',
  title: 'Convolutional Neural Networks for Vision',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Convolutional Neural Networks for Vision' },
    { type: 'text', content:
      'CNNs revolutionized computer vision with hierarchical feature learning. \
Convolutional layers detect edges, textures, and patterns. \
Pooling layers reduce spatial dimensions while preserving features. \
Deep CNN architectures achieve human-level image recognition. \
Transfer learning makes CNNs accessible for various applications.' 
    },
    { type: 'subtitle', content: 'CNN Architectures' },
    { type: 'text', content:
      'LeNet-5: Early CNN for digit recognition. \
AlexNet: Breakthrough architecture for ImageNet competition. \
VGG: Simple architecture with deep layers. \
ResNet: Residual connections enabling very deep networks. \
EfficientNet: Optimized architecture for accuracy and efficiency.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Photos uses CNNs for automatic image organization. \
The system recognizes faces, objects, and scenes in personal photos. \
Users search photos using natural language queries. \
"Show me photos of beaches from last summer" works instantly. \
CNNs transform personal photo collections into searchable memories.' 
    }
  ]
},
'computer-vision-4': {
  id: 'computer-vision-4',
  title: 'Object Detection Algorithms',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Object Detection Algorithms' },
    { type: 'text', content:
      'Object detection identifies and locates multiple objects in images. \
It provides bounding boxes and class labels for each detected object. \
Two-stage detectors (R-CNN family) propose regions then classify. \
One-stage detectors (YOLO, SSD) perform detection in single pass. \
Real-time detection enables applications like autonomous driving.' 
    },
    { type: 'subtitle', content: 'Detection Approaches' },
    { type: 'text', content:
      'R-CNN: Region-based convolutional neural network. \
Fast R-CNN: Improves speed with ROI pooling. \
Faster R-CNN: Uses region proposal network. \
YOLO: "You Only Look Once" for real-time detection. \
SSD: Single Shot MultiBox Detector balances speed and accuracy.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Tesla Autopilot uses object detection for situational awareness. \
The system identifies cars, pedestrians, cyclists, and traffic signs. \
Multiple cameras provide 360-degree environmental perception. \
Real-time detection enables safe autonomous navigation. \
This demonstrates object detection critical safety applications.' 
    }
  ]
},
'computer-vision-5': {
  id: 'computer-vision-5',
  title: 'Image Segmentation Techniques',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Image Segmentation Techniques' },
    { type: 'text', content:
      'Image segmentation partitions images into meaningful regions. \
Semantic segmentation assigns class labels to each pixel. \
Instance segmentation differentiates individual object instances. \
Panoptic segmentation combines semantic and instance approaches. \
Applications include medical imaging, autonomous driving, and photo editing.' 
    },
    { type: 'subtitle', content: 'Segmentation Methods' },
    { type: 'text', content:
      'Thresholding: Simple pixel intensity-based segmentation. \
Edge detection: Identifying boundaries between regions. \
Region growing: Merging similar neighboring pixels. \
Clustering: Grouping pixels by feature similarity. \
Deep learning: U-Net, Mask R-CNN for complex segmentation.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Photoshop uses segmentation for advanced image editing. \
"Select Subject" automatically isolates people or objects. \
The feature uses deep learning to understand image content. \
Users remove backgrounds or apply effects precisely. \
Segmentation powers professional creative workflows.' 
    }
  ]
},
'computer-vision-6': {
  id: 'computer-vision-6',
  title: 'Facial Recognition Systems',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Facial Recognition Systems' },
    { type: 'text', content:
      'Facial recognition identifies or verifies individuals from images or video. \
The process involves detection, alignment, feature extraction, and matching. \
Deep learning models learn discriminative facial features. \
Applications range from security to personalized user experiences. \
Ethical considerations around privacy and bias are crucial.' 
    },
    { type: 'subtitle', content: 'Recognition Steps' },
    { type: 'text', content:
      'Face detection: Locating faces in images. \
Alignment: Normalizing face orientation and scale. \
Feature extraction: Creating face embeddings. \
Matching: Comparing embeddings with database. \
Verification/Identification: Confirming identity or finding matches.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Airport security uses facial recognition for passenger processing. \
Travelers move through checkpoints without showing documents. \
The system matches live faces to passport databases. \
This speeds up security lines while maintaining safety. \
Facial recognition transforms travel efficiency and security.' 
    }
  ]
},
'computer-vision-7': {
  id: 'computer-vision-7',
  title: 'Video Analysis & Action Recognition',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Video Analysis & Action Recognition' },
    { type: 'text', content:
      'Video analysis extends computer vision to temporal sequences. \
Action recognition identifies human activities in video clips. \
Temporal information complements spatial visual features. \
3D CNNs and two-stream networks model spatiotemporal patterns. \
Applications include surveillance, sports analysis, and content moderation.' 
    },
    { type: 'subtitle', content: 'Analysis Techniques' },
    { type: 'text', content:
      'Optical flow: Estimating motion between frames. \
3D convolutions: Processing spatial and temporal dimensions. \
Two-stream networks: Separate spatial and temporal streams. \
Recurrent networks: Modeling temporal dependencies. \
Transformer-based: Attention across space and time.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'NFL uses video analysis for player performance evaluation. \
The system tracks player movements, formations, and plays. \
Coaches analyze tactics and identify improvement areas. \
Broadcasters use analysis for enhanced commentary and graphics. \
Computer vision transforms sports analytics and viewing experiences.' 
    }
  ]
},
'computer-vision-8': {
  id: 'computer-vision-8',
  title: 'Medical Imaging Applications',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Medical Imaging Applications' },
    { type: 'text', content:
      'Computer vision assists in medical diagnosis and treatment planning. \
AI analyzes X-rays, MRIs, CT scans, and ultrasound images. \
Applications include tumor detection, organ segmentation, and disease classification. \
Deep learning achieves expert-level performance in many medical tasks. \
Medical CV improves healthcare accessibility and diagnostic accuracy.' 
    },
    { type: 'subtitle', content: 'Medical CV Tasks' },
    { type: 'text', content:
      'Detection: Identifying abnormalities like tumors or fractures. \
Segmentation: Delineating organs, lesions, or anatomical structures. \
Classification: Diagnosing diseases from medical images. \
Registration: Aligning images from different modalities or times. \
Quantification: Measuring biomarkers or disease progression.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Google Health uses AI for breast cancer detection in mammograms. \
The system analyzes scans with accuracy matching radiologists. \
Early detection improves treatment outcomes and survival rates. \
AI assists in regions with limited medical expertise. \
Computer vision democratizes access to quality healthcare.' 
    }
  ]
},
'computer-vision-9': {
  id: 'computer-vision-9',
  title: 'Augmented Reality & CV',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Augmented Reality & CV' },
    { type: 'text', content:
      'Augmented Reality overlays digital content onto the physical world. \
Computer vision enables AR by understanding real-world environments. \
SLAM (Simultaneous Localization and Mapping) tracks device position. \
Object recognition anchors digital content to physical objects. \
AR applications include gaming, education, retail, and navigation.' 
    },
    { type: 'subtitle', content: 'AR Technologies' },
    { type: 'text', content:
      'Marker-based AR: Uses visual markers for tracking. \
Markerless AR: Relies on feature detection in the environment. \
Projection-based AR: Projects digital content onto surfaces. \
Superimposition-based AR: Replaces or enhances real objects. \
Location-based AR: Ties content to specific geographical locations.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'IKEA Place app uses AR for virtual furniture placement. \
Users see how products would look in their homes before buying. \
Computer vision understands room dimensions and lighting. \
Digital furniture appears realistically in physical spaces. \
AR transforms shopping experiences through visual try-before-you-buy.' 
    }
  ]
},
'computer-vision-10': {
  id: 'computer-vision-10',
  title: 'Autonomous Vehicles & Vision',
  image: 'https://images.unsplash.com/photo-1517077304055-6e89abbf09b0?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Autonomous Vehicles & Vision' },
    { type: 'text', content:
      'Computer vision is crucial for autonomous vehicle perception. \
Cameras provide rich visual information about the driving environment. \
CV systems detect lanes, traffic signs, vehicles, and pedestrians. \
Sensor fusion combines camera data with radar and lidar. \
Reliable perception enables safe autonomous navigation.' 
    },
    { type: 'subtitle', content: 'AV Perception Tasks' },
    { type: 'text', content:
      'Lane detection: Identifying road markings and boundaries. \
Object detection: Recognizing vehicles, pedestrians, cyclists. \
Traffic sign recognition: Understanding regulatory signs. \
Free space detection: Identifying drivable areas. \
Semantic segmentation: Detailed scene understanding.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Waymo autonomous vehicles use computer vision for urban navigation. \
Multiple cameras provide 360-degree environmental awareness. \
The system processes visual data in real-time for decision making. \
Computer vision enables complex urban driving scenarios. \
This technology promises safer, more efficient transportation.' 
    }
  ]
},


//====Ai Ethics====

'ai-ethics-1': {
  id: 'ai-ethics-1',
  title: 'Introduction to AI Ethics',
  image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Introduction to AI Ethics' },
    { type: 'text', content:
      'AI ethics examines moral principles and societal impact of artificial intelligence. \
It addresses fairness, accountability, transparency, and human values in AI systems. \
As AI becomes pervasive, ethical considerations grow increasingly important. \
Ethical AI development balances innovation with social responsibility. \
The field guides responsible creation and deployment of intelligent systems.' 
    },
    { type: 'subtitle', content: 'Core Ethical Principles' },
    { type: 'text', content:
      'Fairness: Avoiding unjust discrimination in AI decisions. \
Transparency: Making AI systems understandable to users. \
Accountability: Ensuring responsibility for AI outcomes. \
Privacy: Protecting personal data used by AI systems. \
Human control: Maintaining meaningful human oversight.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'European Union AI Act establishes legal framework for ethical AI. \
The regulation classifies AI systems by risk levels. \
High-risk applications face strict requirements for transparency and safety. \
This demonstrates government response to AI ethical challenges. \
Regulation shapes how companies develop and deploy AI globally.' 
    }
  ]
},
'ai-ethics-2': {
  id: 'ai-ethics-2',
  title: 'Bias & Fairness in AI Systems',
  image: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Bias & Fairness in AI Systems' },
    { type: 'text', content:
      'AI bias occurs when systems produce unfair outcomes for certain groups. \
Bias can emerge from training data, algorithm design, or system deployment. \
Fairness requires equal treatment and opportunities across demographics. \
Addressing bias involves technical solutions and diverse perspectives. \
Fair AI promotes equitable benefits from technological progress.' 
    },
    { type: 'subtitle', content: 'Bias Sources' },
    { type: 'text', content:
      'Historical bias: Existing societal inequalities in training data. \
Measurement bias: Flawed data collection or labeling processes. \
Aggregation bias: Inappropriate grouping of diverse populations. \
Evaluation bias: Test data not representing real-world distribution. \
Deployment bias: Mismatch between design context and actual use.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Amazon discontinued AI recruiting tool that showed gender bias. \
The system learned historical hiring patterns favoring male candidates. \
Technical fixes failed to eliminate embedded discrimination. \
This case highlighted challenges in debiasing real-world AI. \
Companies now prioritize fairness in HR technology development.' 
    }
  ]
},
'ai-ethics-3': {
  id: 'ai-ethics-3',
  title: 'Transparency & Explainable AI',
  image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Transparency & Explainable AI' },
    { type: 'text', content:
      'Explainable AI (XAI) makes machine learning decisions understandable to humans. \
Transparency builds trust and enables accountability for AI outcomes. \
Complex models like deep neural networks often act as "black boxes." \
XAI techniques reveal how models arrive at specific decisions. \
Explainability is crucial for high-stakes applications like healthcare and finance.' 
    },
    { type: 'subtitle', content: 'XAI Approaches' },
    { type: 'text', content:
      'Interpretable models: Using inherently understandable algorithms. \
Post-hoc explanation: Analyzing trained model behavior. \
Local explanations: Understanding individual predictions. \
Global explanations: Understanding overall model behavior. \
Counterfactual explanations: Showing how changes affect outcomes.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Banks use explainable AI for credit decision transparency. \
Regulations require explaining loan denials to applicants. \
XAI systems show which factors most influenced decisions. \
This complies with fair lending laws and builds customer trust. \
Explainability transforms opaque algorithms into accountable tools.' 
    }
  ]
},
'ai-ethics-4': {
  id: 'ai-ethics-4',
  title: 'Privacy in the Age of AI',
  image: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Privacy in the Age of AI' },
    { type: 'text', content:
      'AI systems often require large amounts of personal data for training. \
Privacy-preserving techniques protect individual information while enabling innovation. \
Data minimization, anonymization, and encryption safeguard personal information. \
Federated learning trains models without centralizing raw data. \
Balancing data utility with privacy protection is an ongoing challenge.' 
    },
    { type: 'subtitle', content: 'Privacy Techniques' },
    { type: 'text', content:
      'Differential privacy: Adding mathematical noise to protect individuals. \
Federated learning: Training across decentralized devices. \
Homomorphic encryption: Computing on encrypted data. \
Synthetic data: Generating artificial datasets preserving patterns. \
Access controls: Limiting data use to authorized purposes.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Apple uses differential privacy for iOS data collection. \
The system learns aggregate patterns without accessing individual data. \
User privacy remains protected while improving services. \
This approach enables Siri improvements without compromising privacy. \
Apple demonstrates privacy-preserving AI at massive scale.' 
    }
  ]
},
'ai-ethics-5': {
  id: 'ai-ethics-5',
  title: 'Accountability & Responsibility',
  image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Accountability & Responsibility' },
    { type: 'text', content:
      'AI accountability ensures clear responsibility for system outcomes. \
When AI causes harm, determining liability can be complex. \
Responsibility spans developers, deployers, users, and regulators. \
Accountability mechanisms include auditing, monitoring, and redress. \
Clear accountability frameworks build public trust in AI systems.' 
    },
    { type: 'subtitle', content: 'Accountability Mechanisms' },
    { type: 'text', content:
      'Audit trails: Recording system decisions and inputs. \
Impact assessments: Evaluating potential harms before deployment. \
Redress procedures: Correcting errors and compensating harms. \
Oversight boards: Independent review of AI systems. \
Certification standards: Third-party validation of AI safety.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Uber suspended self-driving tests after fatal pedestrian accident. \
The incident raised questions about corporate responsibility. \
Investigations examined technical failures and safety protocols. \
This case highlighted accountability challenges in autonomous systems. \
Companies now implement more rigorous safety measures for AI testing.' 
    }
  ]
},
'ai-ethics-6': {
  id: 'ai-ethics-6',
  title: 'AI Safety & Alignment',
  image: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60',
  sections: [
    { type: 'title', content: 'AI Safety & Alignment' },
    { type: 'text', content:
      'AI safety ensures systems behave as intended without causing harm. \
Alignment research focuses on making AI goals match human values. \
As AI systems become more capable, safety considerations grow critical. \
Techniques include value learning, robustness testing, and fail-safes. \
Proactive safety measures prevent unintended consequences.' 
    },
    { type: 'subtitle', content: 'Safety Approaches' },
    { type: 'text', content:
      'Robustness: Ensuring reliable performance across conditions. \
Interpretability: Understanding system decision-making. \
Verification: Mathematically proving system properties. \
Monitoring: Detecting unsafe behavior during operation. \
Corrigibility: Allowing safe modification of deployed systems.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'OpenAI uses reinforcement learning from human feedback (RLHF). \
This technique aligns language models with human preferences. \
Human trainers rate model outputs to guide learning. \
Alignment reduces harmful, untruthful, or biased responses. \
Safety research makes powerful AI systems more beneficial and controllable.' 
    }
  ]
},
'ai-ethics-7': {
  id: 'ai-ethics-7',
  title: 'AI Governance & Regulation',
  image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60',
  sections: [
    { type: 'title', content: 'AI Governance & Regulation' },
    { type: 'text', content:
      'AI governance establishes frameworks for responsible development and use. \
Regulation addresses risks while encouraging innovation and economic growth. \
International cooperation harmonizes standards across borders. \
Multi-stakeholder approaches involve government, industry, academia, and civil society. \
Effective governance balances innovation with public interest protection.' 
    },
    { type: 'subtitle', content: 'Governance Elements' },
    { type: 'text', content:
      'Risk-based regulation: Stricter rules for higher-risk applications. \
Standards development: Technical specifications for safety and interoperability. \
Certification schemes: Independent verification of compliance. \
Oversight bodies: Institutions monitoring AI development and deployment. \
International agreements: Coordinating approaches across countries.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'GDPR includes provisions for automated decision-making. \
The regulation gives individuals right to explanation of algorithmic decisions. \
Companies must implement data protection by design and default. \
This European framework influences global AI development practices. \
GDPR demonstrates how regulation shapes technological innovation.' 
    }
  ]
},
'ai-ethics-8': {
  id: 'ai-ethics-8',
  title: 'AI & Employment Impacts',
  image: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60',
  sections: [
    { type: 'title', content: 'AI & Employment Impacts' },
    { type: 'text', content:
      'AI automation transforms labor markets and job requirements. \
Some tasks become automated while new roles and industries emerge. \
Workforce transitions require reskilling and social support systems. \
Ethical considerations include fair transition and inclusive growth. \
Proactive policies can maximize benefits while minimizing disruption.' 
    },
    { type: 'subtitle', content: 'Employment Strategies' },
    { type: 'text', content:
      'Reskilling programs: Training workers for AI-augmented roles. \
Lifelong learning: Continuous skill development throughout careers. \
Social safety nets: Supporting workers during transitions. \
Job quality focus: Ensuring new jobs offer decent wages and conditions. \
Inclusive design: Creating AI that complements diverse workers.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Amazon invests $700 million in employee upskilling programs. \
The initiative prepares workers for technical roles in AI era. \
Training includes software development, data science, and cloud computing. \
This corporate responsibility approach addresses automation impacts. \
Companies increasingly recognize workforce transition obligations.' 
    }
  ]
},
'ai-ethics-9': {
  id: 'ai-ethics-9',
  title: 'AI for Social Good',
  image: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1000&q=60',
  sections: [
    { type: 'title', content: 'AI for Social Good' },
    { type: 'text', content:
      'AI can address pressing societal challenges and promote human welfare. \
Applications include healthcare accessibility, environmental protection, and education. \
Ethical deployment prioritizes underserved communities and global equity. \
Multidisciplinary collaboration combines technical and domain expertise. \
AI for social good demonstrates technology positive potential.' 
    },
    { type: 'subtitle', content: 'Social Good Applications' },
    { type: 'text', content:
      'Healthcare: Disease diagnosis in resource-limited settings. \
Environment: Climate modeling and conservation monitoring. \
Education: Personalized learning for diverse student needs. \
Agriculture: Crop monitoring and yield optimization. \
Humanitarian aid: Disaster response and resource allocation.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'Microsoft AI for Earth supports environmental sustainability projects. \
The program provides cloud and AI tools to conservation organizations. \
Applications include species monitoring, deforestation tracking, and water management. \
This initiative demonstrates corporate commitment to social responsibility. \
AI becomes force multiplier for environmental protection efforts.' 
    }
  ]
},
'ai-ethics-10': {
  id: 'ai-ethics-10',
  title: 'Future Ethical Challenges',
  image: 'https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?w=1000&q=60',
  sections: [
    { type: 'title', content: 'Future Ethical Challenges' },
    { type: 'text', content:
      'Emerging AI capabilities raise novel ethical questions and concerns. \
Advanced systems may exhibit unprecedented autonomy and influence. \
Societal preparation requires anticipatory governance and public engagement. \
Long-term considerations include existential risks and value lock-in. \
Ongoing ethical reflection guides responsible technological evolution.' 
    },
    { type: 'subtitle', content: 'Future Considerations' },
    { type: 'text', content:
      'Artificial general intelligence: Systems with human-like capabilities. \
Brain-computer interfaces: Merging biological and artificial intelligence. \
Autonomous weapons: Military applications without human control. \
Digital persons: AI entities with legal rights and responsibilities. \
Value alignment: Ensuring advanced AI shares human ethical priorities.' 
    },
    { type: 'image', content: 'https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1000&q=60' },
    { type: 'subtitle', content: 'Real-Time Example' },
    { type: 'text', content:
      'United Nations discusses lethal autonomous weapons systems (LAWS). \
International debate examines ethical and legal implications. \
Some countries advocate for preemptive bans on killer robots. \
This demonstrates global governance addressing emerging AI risks. \
International cooperation shapes responsible development of powerful technologies.' 
    }
  ]
},

//==== 

};

export default data;
