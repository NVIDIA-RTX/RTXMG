/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#include <donut/app/ApplicationBase.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <nvrhi/utils.h>

#ifdef _WIN32
#include <Windows.h>
#include <DbgHelp.h>
#include <cstdio>
#include <cstdlib>
#include <crtdbg.h>
#pragma comment(lib, "DbgHelp.lib")

// Redirect CRT asserts/errors to stderr and then crash so the SEH handler fires.
static int CrtReportHook(int reportType, char* message, int* /*returnValue*/)
{
    if (reportType == _CRT_ASSERT || reportType == _CRT_ERROR)
    {
        const char* tag = (reportType == _CRT_ASSERT) ? "ASSERT" : "ERROR";
        fprintf(stderr, "\n=== CRT %s ===\n%s\n", tag, message ? message : "(no message)");
        fflush(stderr);
        // Raise a structured exception so UnhandledExceptionHandler can print a callstack.
        RaiseException(EXCEPTION_ACCESS_VIOLATION, 0, 0, nullptr);
    }
    return FALSE; // use default handling for _CRT_WARN
}

static LONG WINAPI UnhandledExceptionHandler(EXCEPTION_POINTERS* pExceptionInfo)
{
    HANDLE hProcess = GetCurrentProcess();

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(hProcess, nullptr, TRUE);

    EXCEPTION_RECORD* pRecord = pExceptionInfo->ExceptionRecord;

    // Copy context so StackWalk64 modifications don't corrupt the original
    CONTEXT context = *pExceptionInfo->ContextRecord;

    fprintf(stderr, "\n=== CRASH: Unhandled Exception ===\n");
    fprintf(stderr, "Exception Code:    0x%08lX\n", pRecord->ExceptionCode);
    fprintf(stderr, "Exception Address: %p\n", pRecord->ExceptionAddress);

    STACKFRAME64 frame = {};
#if defined(_M_ARM64)
    const DWORD machineType = IMAGE_FILE_MACHINE_ARM64;
    frame.AddrPC.Offset    = context.Pc;
    frame.AddrFrame.Offset = context.Fp;
    frame.AddrStack.Offset = context.Sp;
#else
    const DWORD machineType = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset    = context.Rip;
    frame.AddrFrame.Offset = context.Rbp;
    frame.AddrStack.Offset = context.Rsp;
#endif
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Mode   = AddrModeFlat;

    char symBuf[sizeof(SYMBOL_INFO) + MAX_SYM_NAME];
    SYMBOL_INFO* pSym = reinterpret_cast<SYMBOL_INFO*>(symBuf);
    pSym->SizeOfStruct = sizeof(SYMBOL_INFO);
    pSym->MaxNameLen   = MAX_SYM_NAME;

    IMAGEHLP_LINE64 line = {};
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

    fprintf(stderr, "\nCallstack:\n");

    HANDLE hThread = GetCurrentThread();
    for (int i = 0; i < 64; ++i)
    {
        if (!StackWalk64(machineType, hProcess, hThread, &frame,
                &context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)
            || frame.AddrPC.Offset == 0)
            break;

        fprintf(stderr, "  [%2d] 0x%016llX", i, static_cast<unsigned long long>(frame.AddrPC.Offset));

        DWORD64 symDisp = 0;
        if (SymFromAddr(hProcess, frame.AddrPC.Offset, &symDisp, pSym))
        {
            fprintf(stderr, "  %s + 0x%llX", pSym->Name, static_cast<unsigned long long>(symDisp));

            DWORD lineDisp = 0;
            if (SymGetLineFromAddr64(hProcess, frame.AddrPC.Offset, &lineDisp, &line))
                fprintf(stderr, "  (%s:%lu)", line.FileName, line.LineNumber);
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "==================================\n\n");
    fflush(stderr);

    SymCleanup(hProcess);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif // _WIN32

extern "C" {

#ifdef DONUT_D3D_AGILITY_SDK_ENABLED
    _declspec(dllexport) extern const UINT D3D12SDKVersion = DONUT_D3D_AGILITY_SDK_VERSION;
    _declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
#endif

    _declspec(dllexport) DWORD NvOptimusEnablement = 0x0000001;
}

#include "rtxmg_demo_app.h"
#include "gui.h"

using namespace donut;

class UIScreenshotPass : public donut::app::IRenderPass
{
    RTXMGDemoApp& m_app;
public:
    UIScreenshotPass(donut::app::DeviceManager* dm, RTXMGDemoApp& app)
        : IRenderPass(dm), m_app(app) {}
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        m_app.CaptureScreenshotWithUI(framebuffer);
    }
};

int main(int argc, const char** argv)
{
#ifdef _WIN32
    SetUnhandledExceptionFilter(UnhandledExceptionHandler);
#ifdef _DEBUG
    _CrtSetReportHook(CrtReportHook);
#endif
#endif

    donut::log::ConsoleApplicationMode();

    // Disable message-box dialogs when running headless (-nf / --nframes),
    // so automated runs and CI harnesses don't block on error popups.
    bool headless = false;
    for (int i = 1; i < argc; ++i)
    {
        // --shot-list is headless too, and deliberately ignores -nf, so the
        // harness' golden scenarios pass no frame count at all.
        if (strcmp(argv[i], "-nf") == 0 || strcmp(argv[i], "--nframes") == 0 ||
            strcmp(argv[i], "--shot-list") == 0)
        {
            headless = true;
            break;
        }
    }
    donut::log::EnableOutputToMessageBox(!headless);

    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(argc, argv);

#if !DONUT_WITH_DX12
    if (api == nvrhi::GraphicsAPI::D3D12)
    {
        donut::log::fatal("This demo supports D3D12 but needs to be compiled with DONUT_WITH_DX12 enabled in cmake");
    }
#endif
#if !DONUT_WITH_VULKAN
    if (api == nvrhi::GraphicsAPI::VULKAN)
    {
        donut::log::fatal("This demo supports Vulkan but needs to be compiled with DONUT_WITH_VULKAN enabled in cmake");
    }
#endif
    if (api == nvrhi::GraphicsAPI::D3D11)
    {
        donut::log::fatal("This demo only supports D3D12 or Vulkan");
    }

    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

#if defined(_M_ARM64) || defined(__aarch64__)
    const char* arch = "arm64";
#else
    const char* arch = "x64";
#endif
    std::string title = "RTX Mega Geometry " RTXMG_VERSION + std::string(api == nvrhi::GraphicsAPI::D3D12 ? " (D3D12 " : " (VULKAN ") + arch + ")";

    try 
    {
        {
            RTXMGDemoApp app(deviceManager, title, argc, argv);
            UserInterface gui(app);
            UIScreenshotPass uiScreenshot(deviceManager, app);
            if (app.Init() && gui.CustomInit(app.GetRenderer().GetShaderFactory()))
            {
                deviceManager->AddRenderPassToBack(&app);
                deviceManager->AddRenderPassToBack(&gui);
                deviceManager->AddRenderPassToBack(&uiScreenshot);
                deviceManager->RunMessageLoop();
                deviceManager->RemoveRenderPass(&uiScreenshot);
                deviceManager->RemoveRenderPass(&gui);
                deviceManager->RemoveRenderPass(&app);
            }

            Profiler::Terminate();
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