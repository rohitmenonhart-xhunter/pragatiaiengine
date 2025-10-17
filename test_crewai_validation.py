#!/usr/bin/env python3
"""
Test the CrewAI Multi-Agent Validation System
Test with a real startup idea to validate 109+ agent orchestration
"""

import os
import sys
import time
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Set environment for testing
os.environ['PYTHONPATH'] = str(app_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Disable CrewAI tracing prompts for testing
os.environ['CREWAI_TRACING_ENABLED'] = 'false'
# Set logging level for more detailed output
os.environ['CREWAI_LOG_LEVEL'] = 'DEBUG'

def test_crewai_system():
    """Test the complete CrewAI validation system"""
    print("🚀 Testing Pragati CrewAI Multi-Agent Validation System")
    print("=" * 60)
    
    try:
        # Import and test the integration
        from crew_ai_integration import PragatiCrewAIValidator
        
        print("✅ Successfully imported PragatiCrewAIValidator")
        
        # Initialize the validator (this creates all 109+ agents)
        print("\n🔧 Initializing validation system...")
        start_time = time.time()
        
        validator = PragatiCrewAIValidator()
        
        init_time = time.time() - start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")
        print(f"🤖 Total agents created: {len(validator.agents)}")
        
        # Print agent distribution
        cluster_counts = validator.agent_factory.get_agent_count_by_cluster()
        print("\n📊 Agent Distribution:")
        for cluster, count in cluster_counts.items():
            print(f"  • {cluster}: {count} agents")
        
        # Test idea validation
        print("\n🧪 Testing Idea Validation...")
        print("-" * 40)
        
        test_idea = {
            "name": "EcoFarm AI - Smart Sustainable Agriculture Platform",
            "concept": """
            EcoFarm AI is an integrated IoT and AI-powered platform specifically designed for Indian farmers 
            to transition to sustainable agriculture practices. The platform combines:
            
            1. **Smart Monitoring**: IoT sensors for soil health, water levels, weather patterns, and crop growth
            2. **AI-Powered Insights**: Machine learning algorithms trained on Indian agricultural data to provide 
               personalized recommendations for crop rotation, organic fertilizer usage, and pest management
            3. **Marketplace Integration**: Direct-to-consumer marketplace connecting farmers with urban buyers 
               interested in organic produce, eliminating middlemen
            4. **Financial Services**: Integration with government schemes, micro-loans, and crop insurance 
               specifically for sustainable farming practices
            5. **Community Platform**: Knowledge sharing between farmers, expert consultations, and best practices 
               for sustainable agriculture
            
            Target Market: Small to medium Indian farmers (1-10 acres) looking to increase yield while 
            adopting environmentally sustainable practices. Initial focus on Maharashtra, Punjab, and Karnataka.
            
            Revenue Model: 
            - SaaS subscription for platform access (₹500-2000/month based on farm size)
            - Commission on marketplace transactions (8-12%)
            - Premium consulting services (₹50-100/hour)
            - Data insights for agri-businesses and government agencies
            
            The platform addresses critical issues like declining soil health, water scarcity, climate change 
            adaptation, and farmer income improvement while supporting India's sustainable development goals.
            """
        }
        
        print(f"💡 Test Idea: {test_idea['name']}")
        print(f"📝 Concept length: {len(test_idea['concept'].split())} words")
        
        # Run validation
        validation_start = time.time()
        
        result = validator.validate_idea(
            test_idea['name'], 
            test_idea['concept']
        )
        
        validation_time = time.time() - validation_start
        
        # Display results
        print(f"\n🎯 Validation completed in {validation_time:.2f} seconds")
        print("=" * 60)
        print("📊 VALIDATION RESULTS")
        print("=" * 60)
        
        if result.get('error'):
            print(f"❌ Validation failed: {result['error']}")
            return False
        
        print(f"📈 Overall Score: {result['overall_score']:.2f}/5.0")
        print(f"🏆 Validation Outcome: {result['validation_outcome']}")
        print(f"🤖 Agents Consulted: {result.get('api_calls_made', 'N/A')}")
        print(f"🤝 Consensus Level: {result.get('consensus_level', 0):.1%}")
        
        # Cluster scores
        if 'cluster_scores' in result:
            print(f"\n📊 Cluster Breakdown:")
            for cluster, score in result['cluster_scores'].items():
                emoji = "🟢" if score >= 4.0 else "🟡" if score >= 3.0 else "🔴"
                print(f"  {emoji} {cluster}: {score:.2f}/5.0")
        
        # Collaboration insights
        if 'collaboration_insights' in result:
            print(f"\n🔍 Key Insights:")
            for insight in result['collaboration_insights'][:3]:  # Show top 3 insights
                print(f"  • {insight}")
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        print(f"  • Total Processing Time: {validation_time:.2f} seconds")
        print(f"  • Average Time per Agent: {validation_time/len(validator.agents):.3f} seconds")
        print(f"  • System Efficiency: {'Excellent' if validation_time < 60 else 'Good' if validation_time < 120 else 'Needs Optimization'}")
        
        # Save detailed report
        if 'html_report' in result:
            report_path = "test_validation_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(result['html_report'])
            print(f"📄 Detailed HTML report saved: {report_path}")
        
        print("\n✅ CrewAI Multi-Agent Validation System Test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 Make sure all dependencies are installed:")
        print("   pip install crewai langchain langchain-openai")
        return False
        
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_health():
    """Test system health and configuration"""
    print("\n🏥 System Health Check")
    print("-" * 30)
    
    # Check environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OpenAI API Key: {'*' * 8}{api_key[-4:]}")
    else:
        print("❌ OpenAI API Key: NOT SET")
        return False
    
    # Check dependencies
    try:
        import crewai
        print(f"✅ CrewAI: {crewai.__version__}")
    except ImportError:
        print("❌ CrewAI: NOT INSTALLED")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except ImportError:
        print("❌ LangChain: NOT INSTALLED")
        return False
    
    try:
        import langchain_openai
        print("✅ LangChain OpenAI: Available")
    except ImportError:
        print("❌ LangChain OpenAI: NOT INSTALLED")
        return False
    
    return True


if __name__ == "__main__":
    print("🎯 Pragati CrewAI Multi-Agent Validation Test")
    print("=" * 60)
    
    # Health check first
    if not test_system_health():
        print("\n❌ System health check failed. Please fix issues before testing.")
        sys.exit(1)
    
    # Main test
    success = test_crewai_system()
    
    if success:
        print("\n🎉 All tests passed! The CrewAI system is ready for production.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)
