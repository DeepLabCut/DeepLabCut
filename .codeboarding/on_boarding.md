```mermaid

graph LR

    User_Interface["User Interface"]

    Project_Data_Workflow_Management["Project & Data Workflow Management"]

    Core_Deep_Learning_Engine["Core Deep Learning Engine"]

    Advanced_Analysis_Post_processing["Advanced Analysis & Post-processing"]

    System_Utilities_Benchmarking["System Utilities & Benchmarking"]

    User_Interface -- "initiates workflows and configurations in" --> Project_Data_Workflow_Management

    User_Interface -- "receives and displays final processed results from" --> Advanced_Analysis_Post_processing

    Project_Data_Workflow_Management -- "provides prepared data and model configurations to" --> Core_Deep_Learning_Engine

    Project_Data_Workflow_Management -- "utilizes general services from" --> System_Utilities_Benchmarking

    Core_Deep_Learning_Engine -- "sends raw pose predictions to" --> Advanced_Analysis_Post_processing

    Core_Deep_Learning_Engine -- "utilizes general services from" --> System_Utilities_Benchmarking

    Advanced_Analysis_Post_processing -- "receives raw predictions from" --> Core_Deep_Learning_Engine

    Advanced_Analysis_Post_processing -- "provides processed results to" --> User_Interface

    System_Utilities_Benchmarking -- "provides foundational services to" --> Project_Data_Workflow_Management

    System_Utilities_Benchmarking -- "provides foundational services to" --> Core_Deep_Learning_Engine

    click User_Interface href "https://github.com/DeepLabCut/DeepLabCut/blob/main/.codeboarding//User_Interface.md" "Details"

    click Project_Data_Workflow_Management href "https://github.com/DeepLabCut/DeepLabCut/blob/main/.codeboarding//Project_Data_Workflow_Management.md" "Details"

    click Core_Deep_Learning_Engine href "https://github.com/DeepLabCut/DeepLabCut/blob/main/.codeboarding//Core_Deep_Learning_Engine.md" "Details"

    click Advanced_Analysis_Post_processing href "https://github.com/DeepLabCut/DeepLabCut/blob/main/.codeboarding//Advanced_Analysis_Post_processing.md" "Details"

    click System_Utilities_Benchmarking href "https://github.com/DeepLabCut/DeepLabCut/blob/main/.codeboarding//System_Utilities_Benchmarking.md" "Details"

```



[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Details



The DeepLabCut architecture is designed as a modular, pipeline-driven, and data-centric system, emphasizing a clear separation of concerns. The analysis consolidates the project's functionalities into five core components, facilitating maintainability, scalability, and user-friendliness for deep learning-based computer vision tasks.



### User Interface [[Expand]](./User_Interface.md)

The primary interaction layer for users, providing both a comprehensive graphical interface (GUI) and a command-line interface (CLI) to initiate, manage, and monitor all DeepLabCut workflows.





**Related Classes/Methods**:



- <a href="https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/cli.py#L1-L1" target="_blank" rel="noopener noreferrer">`deeplabcut.cli` (1:1)</a>

- <a href="https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/__main__.py#L1-L1" target="_blank" rel="noopener noreferrer">`deeplabcut.__main__` (1:1)</a>

- `deeplabcut.gui` (1:1)





### Project & Data Workflow Management [[Expand]](./Project_Data_Workflow_Management.md)

Manages the entire project lifecycle, including creating new DeepLabCut projects, handling video files, extracting frames for labeling, organizing datasets, and managing project-specific configurations. It also integrates model loading and configuration.





**Related Classes/Methods**:



- `deeplabcut.create_project` (1:1)

- `deeplabcut.generate_training_dataset` (1:1)

- `deeplabcut.modelzoo` (1:1)





### Core Deep Learning Engine [[Expand]](./Core_Deep_Learning_Engine.md)

The central computational engine responsible for neural network model definition, training, inference (pose prediction), and internal evaluation, abstracting underlying deep learning frameworks (TensorFlow/PyTorch) via a compatibility layer.





**Related Classes/Methods**:



- <a href="https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/compat.py#L1-L1" target="_blank" rel="noopener noreferrer">`deeplabcut.compat` (1:1)</a>

- `deeplabcut.pose_estimation_tensorflow` (1:1)

- `deeplabcut.pose_estimation_pytorch` (1:1)





### Advanced Analysis & Post-processing [[Expand]](./Advanced_Analysis_Post_processing.md)

Refines raw pose estimation outputs by applying filtering, correcting outliers, performing 3D pose reconstruction from 2D estimations, and handling multi-animal tracking functionalities. It also prepares data for final display.





**Related Classes/Methods**:



- `deeplabcut.post_processing` (1:1)

- `deeplabcut.refine_training_dataset` (1:1)

- `deeplabcut.pose_estimation_3d` (1:1)

- `deeplabcut.pose_tracking_pytorch` (1:1)





### System Utilities & Benchmarking [[Expand]](./System_Utilities_Benchmarking.md)

A foundational component providing a comprehensive set of reusable helper functions, common data structures, video I/O, file system interactions, configuration parsing, plotting, and tools for quantitatively assessing model performance.





**Related Classes/Methods**:



- `deeplabcut.utils` (1:1)

- `deeplabcut.core` (1:1)

- `deeplabcut.benchmark` (1:1)









### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)