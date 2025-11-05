# Publishing Notebooks into the Main DLC Cookbook
### Your Recipe Guide to Contributing to the DLC Cookbook

## Introduction
Hey there, DLC enthusiast! 🌟 Ready to sprinkle your magic into the main DLC cookbook? Whether you're introducing a zesty new dish or giving an old one a twist, this guide's got your back. We'll walk you through how to publish a new notebook or spice up an existing one in the DLC cookbook. Let's get cooking! 🍲📘

## Preliminary Checks
### Check Existing Recipes or Tutorials
   - **Search and Review**: Before you start writing a new recipe, go through the existing DLC Jupyter book to ensure there isn't a tutorial or recipe that covers the topic you have in mind.
   - **Expand Existing Content**: If your content is related to an existing topic, like I/O manipulations, consider expanding or refining that section instead of creating an entirely new recipe. This ensures that the Jupyter book remains concise and that related information is found in one place.
      - **Locate and Review**: Navigate to the particular recipe or tutorial you wish to update in the DLC Jupyter book.
      - **Consider Minor vs. Major Changes**: If you're adding a new section or significantly altering the current content, it might be worth noting the changes at the beginning or end of the recipe for clarity.
      - **Maintain Consistency**: Ensure your updates adhere to the current style, tone, and structure of the existing content to maintain a seamless reading experience.


## Structure of a Recipe
   When crafting your recipe, adhere to the following structure:
   - **Introduction**: Begin with an introductory paragraph that highlights the importance and relevance of the recipe. This sets the stage and gives readers context.
    
   - **Examples/Workflow**: Provide step-by-step instructions or a workflow, supported by examples. This makes it easy for readers to understand and follow along.
    
   - **Conclusion**: Conclude with a summary or highlight the key takeaways of your recipe. You can also provide references or further reading.


Now, let's dive into the process of contributing your content to the DLC Jupyter book.
## Steps

1. **Set-up your local environment.** You need `deeplabcut[docs]` installed:
   You can do this by running the following command:
   ```    
   pip install deeplabcut[docs]
   ```

This command installs DeepLabCut along with the dependencies required to build the documentation.

2. **Fork the DLC Repository**:
   - Go to the DeepLabCut GitHub repository: [https://github.com/DeepLabCut/DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)


   - Click on the `Fork` button on the top-right corner of the page. This will create a copy of the repository in your own GitHub account.
3. **Clone your forked repository**:
   - Navigate to your forked repo on GitHub.
   - Click the `Code` button and copy the URL.
   - Clone the repository to your local machine:
   ```
   git clone [REPO_URL]
   ```
4. **Create a new branch**:
   It's a good practice to create a new branch for each new feature or change:
   ```
    cd [YOUR_REPO_DIRECTORY]
    git checkout -b my-new-notebook
   ```
5. **Create a new notebook** or **update an existing one**.
   - **Creating a new notebook**
      - **Choose Your Topic Wisely:** Before you start, make sure your topic fits the DLC Jupyter book's theme and brings value to its readers. A novel topic or a unique twist on an existing topic can be particularly impactful.
      - **Craft with Care:** Remember, your notebook will be a reference for many. Begin with an engaging introduction, followed by well-structured content, and wrap it up with a conclusion.
      - **Interactive Elements:** One of the strengths of Jupyter notebooks is the ability to combine code, visuals, and narrative. Use interactive plots, widgets, or any other tools that enhance the content and make it engaging.
      - **Save Regularly:** Jupyter auto-saves your work, but it's a good habit to manually save your notebook frequently, especially after making significant changes.
      - **Naming Convention:** Name your notebook in a way that reflects its content and is consistent with other notebook titles in the DLC Jupyter book. This makes it easier for readers to understand the topic at a glance.      
   - **Updating an existing notebook** 
      - Navigate to the location of the existing recipe within the directory: 
          ```
          [YOUR_REPO_DIRECTORY]/docs/recipes/
          ```
      - Open the corresponding Jupyter notebook (.ipynb file) you wish to update.
      - Make the necessary changes or additions to the content.
      - Save the notebook once your updates are finalized.
      - Proceed to **Step 6** *(Proofreading)* and **9** *(Testing the documentation)* (skip Steps 7 and 8).
6. **Proofread:** Double check for spelling and grammatical errors by using Jupyter notebook's spellcheck extension called `spellchecker` (or your preferred spell-checker).
   ```
   jupyter nbextension enable spellchecker/main
   ```
   Once installed, restart your notebook, and when you load your notebook again, you will see the incorrectly spelled words highlighted in red.
7. **Add your notebook** to the recipe directory at `[YOUR_REPO_DIRECTORY]/docs/recipes/`

    - Navigate to the appropriate directory where the Jupyter notebooks are stored for the Jupyter book.
    - Add your Jupyter notebook (.ipynb file) to this directory.
    
    To copy via terminal:
    
    - Unix-based OS users
    
      ```
      cp [YOUR_NOTEBOOK_FILENAME].ipynb [YOUR_REPO_DIRECTORY]/docs/recipes
      ```
    
    - WinOS users:
      ```
      copy new_recipe.ipynb [YOUR_REPO_DIRECTORY]\docs\recipes

      ```

8. **Update `[YOUR_REPO_DIRECTORY]/_toc.yml`** by adding under the *Tutorials & Cookbook* section a **new line** containing the path to your notebook. This creates a link to your notebook on the main DLC book sidebar.

    * For example:
      ```      
      - file: docs/recipes/[YOUR_NOTEBOOK_FILENAME]
      ```

9. **Test the documentation:**

    - Build your notebook into the DLC recipe book
      ```
      jupyter book build [YOUR_REPO_DIRECTORY]
      ```
    - Once build is successful, the newly built book can be accessed at `[YOUR_REPO_DIRECTORY]/_build/html/`.
    - Open `index.html` and check whether your notebook was rendered properly and if the links are working.

10. **Commit your changes:**
   When everything is a-okay, commit your changes to your branch. If not, edit your file and go to back to step 1.
    
    ```
    git add [YOUR_NOTEBOOK_FILENAME]
    git commit -m "Added a new notebook about [YOUR_TOPIC]"
    ```

11. **Push your branch to your fork:**

    ```
    git push origin my-new-notebook
    ```


12. **Submit a Pull Request (PR):**

    - Go to your forked repository on GitHub.
    - You'll likely see a message prompting you to create a pull request from your recently pushed branch. Click `Compare & pull request`.
    - Fill out the PR form with a descriptive title and comments describing your notebook. This will help the maintainers understand the context and purpose of your notebook.
    - Click `Create pull request`.

13. **Make Necessary Changes**: The DeepLabCut maintainers will then review your PR and provide feedback. If changes are required, make the necessary changes on your local branch, commit them, and push the branch again. The PR will automatically update.

14. **🎉PR Approval:🎉** Once your PR is approved, the maintainers will merge it into the main repository. Your notebook will then be a part of the DeepLabCut Jupyter book! Yay!

Remember to always check the [DLC contributing guidelines](https://github.com/DeepLabCut/DeepLabCut/blob/main/CONTRIBUTING.md).


## Wrap-Up 🎉
Alright! 🌟 By now, you've got the playbook to jazz up the DeepLabCut Jupyter book. Remember, it's not just about cooking up new recipes but also spicing up the old ones. Dive in, have fun, and let's make this book a flavor-packed feast for all DLC enthusiasts out there. High-five for joining the party! 🙌🎈
