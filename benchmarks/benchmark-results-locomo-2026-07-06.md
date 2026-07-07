# Micro-RAG LoCoMo QA Full Benchmark Results

- Timestamp: `2026-07-06T22:07:13-04:00`
- Model: `gemini-3.1-flash-lite`
- Token counter: `tiktoken` via `tiktoken:encoding:cl100k_base`
- Total Evaluated Questions: `15`
- Total Matches: `7`
- Overall Accuracy: `0.467`
- Avg Injected Context Tokens: `343.3` (Min: `303`, Max: `408`)

## QA Details

| Conv | Question | Expected | Micro-RAG Answer | Tokens | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | What offer does Gina make to Jon regarding social media? | Helping with making content and managing his social media accounts. | Short answer: As of July 21, 2023, Gina is collaborating with Jon to manage social media for his dance studio. | 321 | CORRECT |
| 2 | When did Jon start expanding his studio's social media presence? | April, 2023 | Jon started expanding his dance studio's social media presence by April 3, 2023. | 329 | CORRECT |
| 2 | What do Jon and Gina both have in common? | They lost their jobs and decided to start their own businesses. | Jon and Gina share a mutual passion for dance, a supportive friendship, and a professional collaboration focused on managing and promoting Jon's dance studio. | 330 | MISSED |
| 2 | Why did Jon combine his clothing business with dance? | he is passionate about dance and fashion | Short answer: Jon did not combine his clothing business with dance; it was Gina who launched an online clothing store while also pursuing dance. Jon's business focus is exclusively on his dance studio. | 322 | MISSED |
| 2 | When did Jon start learning marketing and analytics tools? | July, 2023 | Jon started learning marketing and analytics tools on July 9, 2023. | 343 | CORRECT |
| 4 | When did Joanna plan to go over to Nate's and share recipes? | 5 November, 2022. | Joanna planned to visit Nate's house and bring her chocolate and raspberry cake recipes on November 5, 2022. | 303 | CORRECT |
| 4 | What underlying condition might Joanna have based on her allergies? | asthma | Joanna likely has a sensitivity or intolerance to dairy, as she consistently avoids it in her cooking and baking. | 408 | MISSED |
| 4 | What inspired Joanna's new script in July 2022? | Woodhaven's interesting past and people | Joanna's script in July 2022 was inspired by the history of Woodhaven. | 316 | CORRECT |
| 4 | Who did Nate plan to invite to his gaming party in June 2022? | Tournament friends, old friends, teammates | Nate planned to invite his tournament friends, old friends, and teammates. | 329 | CORRECT |
| 4 | What kind of books does Nate enjoy? | Adventures and magic | The provided information does not mention anything about Nate's interest in books. | 387 | MISSED |
| 5 | Which language is Tim learning? | German | Tim is learning German. | 337 | CORRECT |
| 5 | Which popular music composer's tunes does Tim enjoy playing on the piano? | John Williams | Short answer: The provided information does not specify which particular composers Tim plays, only that he is interested in playing classical music, jazz, and film scores on the piano. | 368 | MISSED |
| 5 | In which month's game did John achieve a career-high score in points? | June 2023 | Short answer: July. | 347 | MISSED |
| 5 | How did Tim get introduced to basketball? | Dad signed him up for a local league | The provided context does not state how Tim was introduced to basketball; it only mentions that he owns a basketball signed by LeBron James. | 371 | MISSED |
| 5 | Which Star Wars-related locations would Tim enjoy during his visit to Ireland? | Skellig Michael, Malin Head, Loop Head, Ceann Sibéal, and Brow Head because they are Star Wars filming locations. | Short answer: Based on the provided context, there is no information indicating that Tim has plans to visit Ireland or specific interests in Star Wars-related locations there; his travel interests currently focus on UK castles, New York City, and Harry Potter-related sites. | 338 | MISSED |
