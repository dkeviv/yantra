
November 23

---



1. [ ] Mac dock icon is not showing up. Create an icon for it
2. [X] For each panel allow it to be closed - like preview doesn't have to be alway there
3. [X] Same for the code.
4. [ ] Is the file tree implemented?
5. [ ] Is the git integration implemented?

---



1. [X] UI changes: Terminal output is taking too much space. Make it 3 column design, Files, Chat and Code. Make Chat vertically full lenght. Align terminal with Code column. Move the agent status to the bottom allowing for more space for the file tree.When a panel is closed , should be able to open it from View menu similar to VSCode.
2. [ ] File Tree - can we make it fully functional . When a file is selected, it should show in the code along with file name in the top . Alow mutliple files to be opened and navigated similar to VScode.When clicked on a folder unable to see subfolders or files.
3. [X] Git - Let us use terminal functionality for full Git. Can the agent do all Git functions with MCP? if not can we integrate it?


---



1. [ ] IMplement View Menu and Document the implementations in the UX.md, features.md , project plan , technical_guide.md and session handoff.
3. [X] Allow multiple terminals to be opened like VSCode. The agent should be fully aware of all the terminals. It should not run a command in a terminal without checking if the terminal is avaialble. If there is a command still running in a command then it should not be interrupted. It should use the terminals intelligently
4. [X] Create an optional view to see file dependencies and parameter dependencies. What is the best way to show? A mindmap ??
5. [X] Any view to be seen will show in the code panel space i.e the 3rd column

---

* [X] Yes need to remove the confidence. What does the Agent status show? Is that to show the progress of implementation.
  If that is the case,
* [X] Agent status can show the progress indicator and the current task in progress
* [X] Let us have 4 buttons in that panel to show Features, Decisions, Changes and Plan. User might start with Requirements or specifications file or everything is just chat discussions on what to implement. These should automatically get converted to Features . Decisions should show any critical decisions made during the development and Plan should show all the tasks , indicating which tasks were completed, what is in progress and what is pending and should also show the milestones like MVP, Phase 1 , Phase 2 etc - all based on the chat. Changes should show the change log for track all changes for audit and transparency.
* [X] When the buttons  features, decision, plan , changes are clicked it should be shown in the place of the Files Panel. Any user actions should be clearly be shown in the Plan as a dependency . On clicking the user action, chat should show the instructions for the user.

---
