import unittest
import os
import shutil
import json
from mrag import BeliefStore
from mrag.adapters.skills import (
    import_openai_tools,
    import_mcp_tools,
    import_from_directory
)

class TestSkillsAdapters(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./test_adapter_mrag_data"
        cls.temp_skills_dir = "./test_temp_skills"
        
        # Cleanup
        for d in [cls.test_dir, cls.temp_skills_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
        os.makedirs(cls.temp_skills_dir)
        
        cls.belief_store = BeliefStore(data_dir=cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        for d in [cls.test_dir, cls.temp_skills_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

    def test_import_openai_tools(self):
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_user_profile",
                    "description": "Fetch user profile details.",
                    "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}}
                }
            },
            {
                "type": "retrieval", # Should be ignored
                "retrieval": {}
            }
        ]

        imported = import_openai_tools(openai_tools, self.belief_store)
        self.assertEqual(imported, 1)
        self.assertEqual(import_openai_tools(openai_tools, self.belief_store), 0)
        
        # Verify it exists in store
        belief = self.belief_store.get_belief("tool_fetch_user_profile")
        self.assertIsNotNone(belief)
        self.assertEqual(belief["metadata"]["tool_name"], "fetch_user_profile")
        self.assertIn("Fetch user profile details", belief["content"])

    def test_import_mcp_tools(self):
        mcp_data = {
            "tools": [
                {
                    "name": "add_numbers",
                    "description": "Adds two floats",
                    "inputSchema": {}
                }
            ]
        }

        imported = import_mcp_tools(mcp_data, self.belief_store)
        self.assertEqual(imported, 1)
        self.assertEqual(import_mcp_tools(mcp_data, self.belief_store), 0)
        
        belief = self.belief_store.get_belief("mcp_add_numbers")
        self.assertIsNotNone(belief)
        self.assertIn("Adds two floats", belief["content"])

    def test_import_mcp_tools_accepts_bare_list(self):
        tools_list = [
            {"name": "list_tool", "description": "Passed as a bare list."}
        ]
        imported = import_mcp_tools(tools_list, self.belief_store)
        self.assertEqual(imported, 1)
        self.assertIsNotNone(self.belief_store.get_belief("mcp_list_tool"))

    def test_import_from_directory(self):
        # Create some mock custom tool files (like Hermes / OpenClaw format)
        hermes_tool = {
            "name": "query_database",
            "description": "Queries sql database with a raw query string."
        }
        
        tool_file_path = os.path.join(self.temp_skills_dir, "db_tool.json")
        with open(tool_file_path, "w") as f:
            json.dump(hermes_tool, f)
            
        imported = import_from_directory(self.temp_skills_dir, self.belief_store)
        self.assertEqual(imported, 1)
        self.assertEqual(import_from_directory(self.temp_skills_dir, self.belief_store), 0)
        
        belief = self.belief_store.get_belief("custom_skill_query_database")
        self.assertIsNotNone(belief)
        self.assertIn("Queries sql database", belief["content"])

if __name__ == "__main__":
    unittest.main()
