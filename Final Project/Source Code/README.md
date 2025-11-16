# In order to run the program locally

1. Install dependencies and activate the virtual environment
    ```sh
    $ uv sync
    ```

2. Collect the dataset
    ```bash
    uv run python data_collection_final.py
    ```

3. Run the graphical user interface
    ```bash
    uv run python final_pred.py
    ```
