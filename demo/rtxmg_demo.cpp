/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include <donut/app/ApplicationBase.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>

extern "C" {
#ifdef _WIN32
    #ifdef DONUT_D3D_AGILITY_SDK_ENABLED
        _declspec(dllexport) extern const UINT D3D12SDKVersion = DONUT_D3D_AGILITY_SDK_VERSION;
        _declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
    #endif

        _declspec(dllexport) DWORD NvOptimusEnablement = 0x0000001;
#endif        
}


#include "rtxmg_demo_app.h"
#include "gui.h"

using namespace donut;

const char* UIData::WindowTitle = "RTX Mega Geometry (" RTXMG_VERSION ")";

int main(int argc, const char** argv)
{
    donut::log::ConsoleApplicationMode();
    donut::log::EnableOutputToMessageBox(true);

    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(argc, argv);
    if (api != nvrhi::GraphicsAPI::D3D12)
    {
        donut::log::fatal("This demo requires D3D12");
    }

    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    try 
    {
        RTXMGDemoApp app(deviceManager, argc, argv);
        UserInterface gui(app);
        if (app.Init() && gui.CustomInit(app.GetRenderer().GetShaderFactory()))
        {
            deviceManager->AddRenderPassToBack(&app);
            deviceManager->AddRenderPassToBack(&gui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&gui);
            deviceManager->RemoveRenderPass(&app);
        }
        deviceManager->Shutdown();

        delete deviceManager;
    }
    catch (const std::exception& e)
    {
        donut::log::fatal(e.what());
    }
    return 0;
}

#ifdef WIN32
int WinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPSTR lpCmdLine,
    _In_ int nCmdShow)
{
    return main(__argc, (const char**)__argv);
}
#endif