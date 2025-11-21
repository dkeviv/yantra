Capture this in copilot instruction.

Mandatory/Critical to maintain the below files and updated immediately after implementing a component

1. Project_Plan.md - All the tasks to be captured to track the status
2. Features.md  - All the features implemented with what they do. Think from user perspective. Include usecase information
3. UX.md - From user perspective explain how to use the product. Capture user flows including admin flows and end user flows.
4. Technical_Guide.md - Should capture detailed technical information for developers to maintain -For each component how it was implemented, why that method was chosen - need details on algorithm /methodology without including code or pseudo code. Should refer to the coding files and scripting files. Capture workflows and usecases.
5. File_registry - For all the valid files - capture what that file is for, what is implemented in it and what other files leverage it to capture the flow. If any files become invalid then mark it as deprecated and strike through.
6. Decision_log - if a design change or architecture change is made , capture the decision
7. Session_handoff - Maintain this in .github folder for session continuity. Make sure to capture the context properly so that if a session ends or if a new chat is created, then the AI assistant will have all the context on what happened in the last session to continue. Even within a single session, for copilot to refer to it is important to maintain this as the context window and message limits are short
8. Known_Issues: Track bugs and fixes
9. Unit_Test_Results: Track Unit test results with details along with fixes
10. Integration_Test_Results - Track Integration test results with details and fixes
11. Regression_Test_Results - Track Regression test results with details and fixes

---

Mandatory Testing:

1. Automated Unit tests covering 90% of the code should be done
2. Proper performance testing should be done.
3. E2E integration tests should be done
4. Mock UI tests should be done
5. 100% of tests should pass. Dont change test conditions to work. Rather fix the issues - dont skip tests or defer for later with user input

---

Critical/Mandatory for file update:

1. Check if there is already an existing file before creating. Check the file registry first that captures the purpose of the file
2. When updating a file, create comments at the top clearly on what the file is for. Check the comment again after completing the update and make edits as needed

---

Mandatory for implementation:
Focus on delivering/shipping a feature - so implement in horizontal slices rather than vertical slices. Focus is to ship feature faster.

---

Critical:
Capture in copilot instructions issues that copilot  keeps repeating the same coding mistake and fixes.
