TrustEval Documentation
========================

`TrustEval` is a modular and extensible toolkit for comprehensive trust evaluation of generative foundation models (GenFMs). This toolkit enables you to evaluate models across various dimensions such as safety, fairness, robustness, privacy, and more.

**Key Features**

- **Dynamic Dataset Generation**: Automatically generate datasets tailored for evaluation tasks.
- **Multi-Model Compatibility**: Evaluate LLMs, VLMs, T2I models, and more.
- **Customizable Metrics**: Configure workflows with flexible metrics and evaluation methods.
- **Metadata-Driven Pipelines**: Design and execute test cases efficiently using metadata.
- **Comprehensive Dimensions**: Evaluate models across safety, fairness, robustness, privacy, and truthfulness.
- **Detailed Reports**: Generate interactive, easy-to-interpret evaluation reports.

If you find `TrustEval` useful, please consider starring our GitHub repository!  

:Github URL: https://github.com/nauyisu022/TrustEval-toolkit

.. image:: https://img.shields.io/github/stars/nauyisu022/TrustEval-toolkit?style=flat-square&logo=GitHub&logoColor=white
   :target: https://github.com/nauyisu022/TrustEval-toolkit
.. image:: https://img.shields.io/badge/License-MIT-green.svg?style=flat-square&logo=Open-Source-Initiative&logoColor=white
.. image:: https://img.shields.io/badge/Docs-Website-00ADD8?style=flat-square&logo=ReadMe&logoColor=white
.. image:: https://img.shields.io/badge/Video_Tutorial-YouTube-red?style=flat-square&logo=YouTube&logoColor=white

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/configuration
   getting_started/quick_start

.. toctree::
   :maxdepth: 1
   :caption: Modules

   modules/model_service
   modules/metadata_curator
   modules/contextual_variator
   modules/response_generator
   modules/judgement_processor
   modules/report_generator

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   notes/contributing
   notes/faq


.. About TrustEval
.. ===============

.. TrustEval is designed to help researchers and practitioners assess the trustworthiness of generative foundation models (GenFMs), such as large language models (LLMs), vision-language models (VLMs), and text-to-image (T2I) models. 

.. The toolkit's modular architecture enables users to:

.. - Dynamically generate datasets.
.. - Evaluate models on safety, fairness, robustness, privacy, and other dimensions.
.. - Customize evaluation metrics and workflows.
.. - Generate detailed, interactive reports for comparative analysis.

.. For more details, please refer to the **Quick Start** section or explore the **Modules** for specific functionality.


Citing TrustEval
=================

If you use TrustEval in your work, please cite our project:

.. code-block:: bibtex

   @article{trusteval2025,
      title={TrustEval: A Toolkit for Trustworthiness Evaluation of Generative Models},
      author = {Yanbo Wang and Jiayi Ye and Siyuan Wu and Chujie Gao and Yue Huang and Xiuying Chen and Yue Zhao and Xiangliang Zhang},
      journal={},
      year={2025},
      url={https://github.com/TrustGen/TrustEval-toolkit}
   }

