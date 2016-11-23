#include "stdafx.h"
#include "CppUnitTest.h"
#include "Mouse.h"
#include "Panel.h"

#define EPSILON 0.01f

int testFcn();

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

bool AlmostEqual(float f1, float f2)
{
	return fabs(f1 - f2) < EPSILON*(fabs(f1) + fabs(f2));
}


namespace utest_LBM_GL_CPP
{
//	TEST_CLASS(WindowTest)
//	{
//	public:
//		TEST_METHOD(Construction)
//		{
//			int w = 200;
//			int h = 100;
//			Window myWindow(w, h);
//			std::string myTitle = "Window Title";
//			myWindow.SetTitle(myTitle);
////			Assert::AreEqual(myWindow.GetWidth() , w);
////			Assert::AreEqual(myWindow.GetHeight(), h);
////			Assert::AreEqual(myWindow.GetTitle(), myTitle);
//		}
//		TEST_METHOD(AddFrame)
//		{
//			int w = 200;
//			int h = 100;
//			Window myWindow(w, h);
//			std::string myTitle = "Window Title";
//			myWindow.SetTitle(myTitle);
//			RectInt rect(1, 2, 3, 4);
//			std::string frameTitle("Frame Title");
//			myWindow.CreateFrame(rect,frameTitle);

//			Frame candidateFrame = myWindow.GetFrameByID(0);
//			Frame canonicalFrame(rect, frameTitle);

//			Assert::AreEqual(candidateFrame.GetRectInt().m_x, canonicalFrame.GetRectInt().m_x);
//			Assert::AreEqual(candidateFrame.GetRectInt().m_y, canonicalFrame.GetRectInt().m_y);
//			Assert::AreEqual(candidateFrame.GetRectInt().m_w, canonicalFrame.GetRectInt().m_w);
//			Assert::AreEqual(candidateFrame.GetRectInt().m_h, canonicalFrame.GetRectInt().m_h);
//			Assert::AreEqual(candidateFrame.GetTitle(), canonicalFrame.GetTitle());
//		}
//		TEST_METHOD(AddFrames)
//		{
//			int w = 200;
//			int h = 100;
//			Window myWindow(w, h);
//			std::string myTitle = "Window Title";
//			myWindow.SetTitle(myTitle);
//			RectInt rect1(1, 2, 3, 4);
//			RectInt rect2(2, 3, 4, 5);
//			RectInt rect3(3, 4, 5, 6);
//			std::string frameTitle1("Frame Title1");
//			std::string frameTitle2("Frame Title2");
//			std::string frameTitle3("Frame Title3");
//			myWindow.CreateFrame(rect1,frameTitle1);
//			myWindow.CreateFrame(rect2,frameTitle2);
//			myWindow.CreateFrame(rect3,frameTitle3);

//			Frame candidateFrame = myWindow.GetFrameByID(2);
//			Frame canonicalFrame(rect3, frameTitle3);

//			Assert::AreEqual(candidateFrame.GetRectInt().m_x, canonicalFrame.GetRectInt().m_x);
//			Assert::AreEqual(candidateFrame.GetRectInt().m_y, canonicalFrame.GetRectInt().m_y);
//			Assert::AreEqual(candidateFrame.GetRectInt().m_w, canonicalFrame.GetRectInt().m_w);
//			Assert::AreEqual(candidateFrame.GetRectInt().m_h, canonicalFrame.GetRectInt().m_h);
//			Assert::AreEqual(candidateFrame.GetTitle(), canonicalFrame.GetTitle());
//		}
//	};

//	TEST_CLASS(FrameTest)
//	{
//	public:
//		TEST_METHOD(Construction1)
//		{
//			Frame myFrame(50, 100, 200, 150);
//			Assert::AreEqual(myFrame.GetRectInt().m_x, 50);
//			Assert::AreEqual(myFrame.GetRectInt().m_y,100);
//			Assert::AreEqual(myFrame.GetRectInt().m_w,200);
//			Assert::AreEqual(myFrame.GetRectInt().m_h,150);
//		}
//		TEST_METHOD(Construction2)
//		{
//			std::string title("window title");
//			RectInt rect(50, 100, 200, 150);
//			Frame myFrame(rect, title);
//			Assert::AreEqual(myFrame.GetRectInt().m_x, 50);
//			Assert::AreEqual(myFrame.GetRectInt().m_y,100);
//			Assert::AreEqual(myFrame.GetRectInt().m_w,200);
//			Assert::AreEqual(myFrame.GetRectInt().m_h,150);
//		}
//		TEST_METHOD(SetFrameSizeAndTitle)
//		{
//			RectInt rect(50, 100, 200, 150);
//			RectInt rect2(25, 75, 120, 180);
//			Frame myFrame(rect2);
//			myFrame.SetSize(rect);
//			std::string title("hello");
//			myFrame.SetTitle(title);
//			Assert::AreEqual(myFrame.GetRectInt().m_x, 50);
//			Assert::AreEqual(myFrame.GetRectInt().m_y,100);
//			Assert::AreEqual(myFrame.GetRectInt().m_w,200);
//			Assert::AreEqual(myFrame.GetRectInt().m_h,150);
//			Assert::AreEqual(myFrame.GetTitle(),title);
//		}
//	};


	//// rectInt and rectFloat
	TEST_CLASS(CoordTransformations)
	{
	public:
		TEST_METHOD(intToFloat)
		{
			RectInt rectChild(10, 120, 40, 50);
			RectInt rectParent(30, 35, 100, 200);
			RectFloat rectFloat = RectInt2RectFloat(rectChild, rectParent);
			Assert::IsTrue(AlmostEqual(rectFloat.m_x, -0.8f));
			Assert::IsTrue(AlmostEqual(rectFloat.m_y,  0.2f));
			Assert::IsTrue(AlmostEqual(rectFloat.m_w,  0.8f));
			Assert::IsTrue(AlmostEqual(rectFloat.m_h,  0.5f));
		}

		TEST_METHOD(RectFloatMultiplication)
		{
			RectFloat rect0(-0.25f, -0.2f, 1.4f, 1.2f);
			RectFloat rect1(-1.f, -1.f, 1.f, 1.f);
			RectFloat rect2(0.f, 0.f, 0.5f, 0.5f);
			RectFloat rect3 = rect1*rect2;

			RectFloat rect4 = (rect2*rect1)*rect0;
			RectFloat rect5 = rect2*(rect1*rect0);
			Assert::IsTrue(AlmostEqual(rect3.m_x, -0.5f));
			Assert::IsTrue(AlmostEqual(rect3.m_y, -0.5f));
			Assert::IsTrue(AlmostEqual(rect3.m_w,  0.25f));
			Assert::IsTrue(AlmostEqual(rect3.m_h,  0.25f));
			Assert::IsTrue(rect4 == rect5);
	
		}

		TEST_METHOD(RectFloatIdentity)
		{
			RectFloat rect1(-1.f, -1.f, 1.f, 1.f);
			RectFloat rect2(-1.f, -1.f, 1.f, 1.f);
			//Assert::IsTrue(rect1 == rect2);

		
		}

	};


	TEST_CLASS(Panels)
	{
	public:
		int w = 500;
		int h = 250;
		TEST_METHOD(CreatePanel)
		{
			RectInt rectInt(10, 20, 300, 125);
			RectFloat rectFloat(0.04, 0.16, 0.6, 0.5);

			Panel myPanel = Panel();
			Panel myPanelInt = Panel(rectInt, Panel::DEF_ABS, "my Int Panel");
			Panel myPanelFloat = Panel(rectInt, Panel::DEF_ABS, "my Float Panel");

		}

		TEST_METHOD(CreateSubPanels_Basic)
		{
			RectInt rectInt(10, 20, 1000, 500);
			RectFloat rectFloat(-0.5, -0.9, 0.6, 0.8);
			Panel myPanelInt = Panel(rectInt, Panel::DEF_ABS, "Base Panel");
			myPanelInt.CreateSubPanel(RectInt(100,50,300,200), Panel::DEF_ABS, "absolute subpanel1");
			Assert::IsTrue(myPanelInt.m_subPanels[0]->m_parent->m_rectInt_abs == rectInt);

		}

		TEST_METHOD(CreateSubPanels)
		{
			RectInt rectInt(10, 20, 1000, 500);
			RectFloat rectFloat(-0.5, -0.9, 0.6, 0.8);
			Panel myPanelInt = Panel(rectInt, Panel::DEF_ABS, "Base Panel");
			myPanelInt.CreateSubPanel(RectInt(100,50,300,200), Panel::DEF_ABS, "absolute subpanel1");
			myPanelInt.CreateSubPanel(RectFloat(0.2,-0.9,0.7,1.6), Panel::DEF_REL, "relative subpanel2");
			myPanelInt.m_subPanels[1]->CreateSubPanel(RectFloat(-0.8, -0.5, 1.6, 0.5), Panel::DEF_REL, "relative subpanel of 2");

			Assert::IsTrue(myPanelInt.m_subPanels[0]->m_rectFloat_abs == RectFloat(-0.8, -0.8, 0.6, 0.8));
			Assert::IsTrue(myPanelInt.m_subPanels[1]->m_rectFloat_abs == RectFloat(0.2, -0.9, 0.7, 1.6));
			Assert::IsTrue(myPanelInt.m_subPanels[1]->m_subPanels[0]->m_rectFloat_abs == RectFloat(0.27, -0.5, 0.56, 0.4));
		}

		TEST_METHOD(AbsIntToAbsFloat)
		{
			RectInt rectInt(10, 20, 300, 125);
			//RectFloat rectFloat(-0.93333333, -0.68, 2.0, 2.0);
			RectFloat rectFloat(-1, -1, 2.0, 2.0);
			Panel myPanelInt = Panel(rectInt, Panel::DEF_ABS, "my Int Panel");

			// ensure auto generated absFloat coordinates are correct
			Assert::IsTrue(myPanelInt.m_rectFloat_abs == rectFloat);

			// create subpanel using relative float, andensure absFloat coordinates are correct


			// create subpanel using relative int, andensure absFloat coordinates are correct

			
			// create subpanel using absolute int, andensure absFloat coordinates are correct
		}
	};


	TEST_CLASS(MouseTest)
	{
	public:
		TEST_METHOD(ConstructionAndUpdate)
		{
			Mouse myMouse;
			myMouse.Update(100, 200);
			Assert::AreEqual(myMouse.GetX(),100);
			Assert::AreEqual(myMouse.GetY(),200);
		}
	};


}