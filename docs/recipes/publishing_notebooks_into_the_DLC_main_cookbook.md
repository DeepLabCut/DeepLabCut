# Publishing Notebooks into the Main DLC Cookbook
*Date: 13 June 2023*


## Introduction
Publishing notebooks into the main DLC cookbook can be done in a few easy steps!

## Requirements
To accomplish this, you need `deeplabcut[docs]` installed:

You can do this by running the following command:
    
    pip install deeplabcut[docs]
    
    

## Steps
1. Double check for spelling and grammatical errors (on Grammarly - https://grammarly.com/ or by using Jupyter notebook's spellcheck extension called `spellchecker`).
    ***
    ```
    jupyter nbextension enable spellchecker/main
    ```
    ***
    Once installed, restart your notebook, and when you load your notebook again, you will see the incorrectly spelled words highlighted in red. See example below:

<img src="https://lh3.googleusercontent.com/pw/AIL4fc8RD-cyD-K_-0AHPiMiFVLwBq98_sPNo_DzNZwbEJ1ogS_f0GHVynW_ax8D3AJkNCtPhPQKHUbTPt64_2Up5W8ejsVgdSLf-8jAIzgsPnSwk0zVpiPG-VBgwn7VpxEcqEqor7TJBtlkvbP6Spob1BJyrIri5MqhX3A7IpyDqU2zGlfWnZNW7JWrvQj4ZxrjyBehyaaDJHiL-F5ru7aHqAP1BfXkYGdMqJcqsFs2ntslnGUPifSdEigv2Mrr5ZR_1U9OW5QlfGDKOIR4dacnrtZHKKQH03oUG3rnfQ8UMfu74gim_uqP3AXe1LIL2jlGe-5iSypFlvNuMNqZ7IobPUs66NhHuESf_hFxBhLQSKGER3H7UwCoh8LmrpM3817Fgfgghu7Vmb3OMx0A_vfg5acgvKtHFl7oC0c9Q_CeGPasATIzlFCwWdHF2snbEXAaZhOhCKuSI6CEK-ktoziqoJrqJLX9E38EceqpfwX0VZU8IEKLdshShUVmknH7QRYPAG18EwPy901bG8quOXOUrHdLMbA8aDhS70N5XVCtM58Gt2tJb0ehduNLlTpIySMJ17GVb_I5v8djlxQ7g1DFPbPGWHV9RDoUDhvOrIUaOeU1S3PVaVtt4i7VvFC5hqjQffig7kYge-zoE1nS2HCQ6Q_NnTQAT0HRKg1egQVj-sCMhSQjOPr9JEmy-2QNmUzCfJ_tA6jSWBRwC4VMatkmPYVRg4bzPE02NZNelHBYlDKO1UKnY3UwEr_sFtbghyK2Fe2mi5s3Xs39kp0mBXVxkd_bbnCNmGurRiM5fzrd4razkamDKGyU31oOGC9Wo3yxQqT7IJmnfRphdah1TqAUJftVQqXil9ksSFPOFLTOAL6earFNPAVvwA2Y_n6NETkdukXYcyz8uZBDJAZB0bVmT9rIohI=w1849-h304-s-no?authuser=0"></img>

2. Copy your notebook (ex: `new_recipe.ipynb`) to the recipe directory: `docs/recipes/`
    Via terminal:
    ***
    ```
    cp new_recipe.ipynb path/to/DLC-repo/docs/recipes
    ```
    ***
3. Add the path to `new_recipe.ipynb` under the Tutorials & Cookbook section in the `path/to/DLC-repo/_toc.yml`.
    ***
    ```
      - file: docs/recipes/new_recipe
    ```
    ***
    <img src="https://lh3.googleusercontent.com/pw/AIL4fc_mm8j6SnqYG9mLEEkpSNPk_jpM4p0Zyk5Pmie1c06JtOjbCRELwmfl4CxZx0-dxSF_p4DkXKSSl6comBhjgsLHMnAxYyYfgUVVYz9mwL-37Ol6sfdq4QYcjcs5NLqIP2SWZduyvjmoHpj5miJi0c773SgU_RUqCXqCJNFAHr5VxpM8Fz80K2hZHtNkh5hxBc1g4eyGGbM0wYVhQobEUnyveWzJ8lN-J4GMUPHUqh0Bf05Wjc1zdyDtxORzVZ-kIdwmZLKapo-FWG7zLAEa6BpdZpCluGFnnBPnPWtJnpaS4RvST7QBJLNpjiqgR-6NPBUcrHabo7G0RypI-KGJ9ayfw41GZ9U6GS0zJmUEHtAqyhuWyyyDV8hK4q2dy5el9TZ6OTiQ_wBJHfpzLmMiWzORfcDetXS3CCtLvnpT3kGUH8HWqsXHASVEvgNYlMvYObt7X9c7PcX51otfBheuKsQSfxVvPCrvRoVYFtYP-U1GCH8phL3D3kKYLOWdbIoI2NotIaMXp8HXNAle-AFE1snw964ISILKHFhrlbI_EEJQOgk0FmwScmBYLc37DmgjmMNNp5KZaugHJn48tNU4A108bCMfJaB95e_AeloBDIjtzKhkDcPpDRNC_FvUkKQSrjk0QSdM27WFMNH1ph0RW6tZRlZL3e_15ptt2Nowt_M4jbnINTyROKbQA6Xh8VCAh1twZhhFIj6LCURPXtmoMBXYst_V9u-W0cRIgNVV2jZuV2-uoCOpSJPW5z-cSiodGnes7XyqhCKysAAWpepsZPIZHWGwtjq56_txyoMBEV3ce0NaHx0lnrnzEq3-FSV0BAPCJR5WPeugNhDb6BypeFpYMRmEaFS75l7ChVqdcml8kEiP7mf-B10yyojenCn7hOz4AGAYROyxWUFEre1SobZY9NU=w1038-h541-s-no?authuser=0"></img>
4.  Build your notebook into the DLC recipe book
    ***
    ```
        jupyter book build path/to/DLC-repo
    ```
    ***
    The build log should look like below:
    <img src="https://lh3.googleusercontent.com/pw/AIL4fc97W2rW08Mgc2I8vL8qi-Oo_a9klpK0Cly-huhENBe7f8BhXOZzASkbkAe2MLxUkRSyn0SvQm5Dc48iHNCxVnTqONtamUwLKXBtRzkHFpNNcGJwqJDuBKZFuA07RKpHm88f-W59Z7aVpYvYOkllUZfkGpj86q6DcYAX9CAuOdDhcnUcuMoLnheOpKtGU12BMJionXoLUhhpumvf7H2G6WL8yPg5PG8_prrjcKLRLznH5K9LqqwoiHKllsdyqqWAr9yycnnhjzjtczbZF2N1aiBt866uvRnYLO1QWyBD8ThwPKD4v2bqIW3DNjhVzcFOKQZFSqDNMeurk2MLfErxnD5gVAe_0n4ULMdQmXGiG5exEYcpINChDelAvyrxXwV2k1O3mEP6u5aYiqyoy4qFq7XtcH2ERnwgwfCWZLXdbBmgQl9kWJxbpPbOhJ9GZJiHswvVzxhEQrtSTLGGMCbF0W5QJYCv5o0Rdv1FYhGDwcgzSNl1W_1iMFoqr7KtUsTDObIZfrpttwBOA5W_-ZD8mlN8G210-ChluZ5poZEjgulZ6K5QcjblRtJCZdxGXYGLH5D1IMd5QCvMEQfmlkmuexzbb45ZicOqlIWdRngfltWmt_WCE86xTN5pVNV2iEnWQySyZ4FCTTwKfbWwsjaxbV0CUcecJFe4pTkfaCGJ5lpFVFIZDh96SBkMW2ma4sOmelHQMwKw5Hvt9THI8V1OlOtukT7xdf6Daal5OcGJ-RWmbwrHJxWFnS974I9FskWoateZw-P_8u8Qi7astUdbPR8zIOUt6iCgTVYlHfzKKhhJG54LL946tqDLVTuGFXo8rci58vQHk3Z8jhLAP0rAmqaILRoYrnsZI9GyuzkNvkwDM7jzstPAjkBy_9PTfJC04Ycw1nNOS5OnQEodxztU_0XGvQ8=w1387-h405-s-no?authuser=0"></img>

5. Test locally by checking the `index.html` file in `path/to/DLC-repo/_build/html/`

6. When everything is a-okay, commit to Git. If not, edit your file and go to back to step 1.
    
    **`git status`** to check the local changes in your current project
    ```
    git status
    ```
    **`git add`** to add your file/s to the commit bin
    ```
    git add [filename]
    ```
    
    **`git commit`** to add your file/s to the commit bin
    ```
    git commit -m "commit message here; make it descriptive!"
    ```
    **`git pull`** or **`git rebase`** to update your local copy from the main branch.
    ```
    git rebase
    ```
    **`git push`** to push your changes to the main branch.
    ```
    git push
    ```
7. When everything's clear, confirm your pull request on the Git website: https://github.com/DeepLabCut/DeepLabCut

## All done!
