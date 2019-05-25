use rendy::{
    command::Families, factory::Config, factory::Factory, graph::present::PresentNode,
    graph::render::*, graph::Graph, graph::GraphBuilder, wsi::Surface,
};

mod gui;
mod tex;
mod triangle;

use tex::SpriteGraphicsPipeline;
use tex::SpriteGraphicsPipelineDesc;
use triangle::TriangleRenderPipeline;
use triangle::TriangleRenderPipelineDesc;

use core::borrow::BorrowMut;
use gui::ImguiRenderPipeline;
use shred::Resources;
use std::alloc::handle_alloc_error;
use std::cell::RefCell;
use winit::{EventsLoop, WindowBuilder};

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

fn init_imgui(window: &winit::Window) -> imgui::ImGui {
    use imgui::{FontGlyphRange, ImFontConfig, ImGui, ImVec4};

    let mut imgui = ImGui::init();
    {
        // Fix incorrect colors with sRGB framebuffer
        fn imgui_gamma_to_linear(col: ImVec4) -> ImVec4 {
            let x = col.x.powf(2.2);
            let y = col.y.powf(2.2);
            let z = col.z.powf(2.2);
            let w = 1.0 - (1.0 - col.w).powf(2.2);
            ImVec4::new(x, y, z, w)
        }

        let style = imgui.style_mut();
        for col in 0..style.colors.len() {
            style.colors[col] = imgui_gamma_to_linear(style.colors[col]);
        }
    }
    imgui.set_ini_filename(None);

    // In the examples we only use integer DPI factors, because the UI can get very blurry
    // otherwise. This might or might not be what you want in a real application.
    let hidpi_factor = window.get_hidpi_factor().round();

    let font_size = (13.0 * hidpi_factor) as f32;

    imgui.fonts().add_default_font_with_config(
        ImFontConfig::new()
            .oversample_h(1)
            .pixel_snap_h(true)
            .size_pixels(font_size),
    );

    imgui.fonts().add_font_with_config(
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/mplus-1p-regular.ttf"
        )),
        ImFontConfig::new()
            .merge_mode(true)
            .oversample_h(1)
            .pixel_snap_h(true)
            .size_pixels(font_size)
            .rasterizer_multiply(1.75),
        &FontGlyphRange::default(),
    );

    imgui.set_font_global_scale((1.0 / hidpi_factor) as f32);

    return imgui;
}

fn init_render_graph(
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    surface: Surface<Backend>,
    aux: &Resources,
) -> Graph<Backend, Resources> {
    // GraphBuilder gives us a declarative interface for describing what/how to render. Using this
    // structure rather than directly making calls on a GPU backend means much of the error
    // handling and recovery (such as the device being lost) are automatically handled
    let mut graph_builder = GraphBuilder::<Backend, Resources>::new();

    // The frame starts with a cleared color buffer
    let color = graph_builder.create_image(
        surface.kind(),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [0.9, 0.7, 0.7, 1.0].into(),
        )),
    );

    let depth = graph_builder.create_image(
        surface.kind(),
        1,
        gfx_hal::format::Format::D16Unorm,
        Some(gfx_hal::command::ClearValue::DepthStencil(
            gfx_hal::command::ClearDepthStencil(1.0, 0),
        )),
    );

    let pass0 = graph_builder.add_node(
        SpriteGraphicsPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    // Render a triangle (this turns the pipeline into a subpass, provides the color buffer,
    // then turns it into a pass
    let pass1 = graph_builder.add_node(
        TriangleRenderPipeline::builder()
            .with_dependency(pass0)
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    // Render imgui
    let pass2 = graph_builder.add_node(
        ImguiRenderPipeline::builder()
            .with_dependency(pass1)
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    // Then present the pass to the screen
    graph_builder.add_node(PresentNode::builder(&factory, surface, color).with_dependency(pass2));

    return graph_builder.build(factory, families, aux).unwrap();
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn run(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    graph: &mut Graph<Backend, Resources>,
    window: std::sync::Arc<winit::Window>,
    resources: &mut Resources,
) -> Result<(), failure::Error> {
    loop {
        let mut is_close_requested = false;

        factory.maintain(families);

        event_loop.poll_events(|evt| {
            use winit::Event;
            use winit::WindowEvent;

            match evt {
                // Close if the window is killed
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => is_close_requested = true,

                // Close if the escape key is hit
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            input:
                                winit::KeyboardInput {
                                    virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        },
                    ..
                } => is_close_requested = true,

                //Process keyboard input
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { input, .. },
                    ..
                } => {
                    log::debug!("print {:?}", input);
                    //input_listener.handle_keyboard_event(&input);
                }

                _ => log::debug!("Received winit event"),
            }
        });

        let window_size = window.get_inner_size();
        let window_size = window_size.unwrap();

        let mut imgui = resources.fetch_mut::<imgui::ImGui>();
        let mut ui = imgui.frame(
            imgui::FrameSize {
                logical_size: (window_size.width, window_size.height),
                hidpi_factor: window.get_hidpi_factor().round(),
            },
            0.0,
        );

        imgui_ui_render(&window, &mut ui);

        graph.run(factory, families, resources);

        if is_close_requested {
            break;
        }
    }

    Ok(())
}

fn imgui_ui_render(window: &winit::Window, ui: &mut imgui::Ui) {
    use imgui::im_str;
    use imgui::ImGuiCond;

    ui.window(im_str!("Hello world"))
        .size((300.0, 100.0), ImGuiCond::FirstUseEver)
        .build(|| {
            ui.text(im_str!("Hello world!"));
            ui.text(im_str!("こんにちは世界！"));
            ui.text(im_str!("This...is...imgui-rs!"));
            ui.separator();
            let mouse_pos = ui.imgui().mouse_pos();
            ui.text(im_str!(
                "Mouse Position: ({:.1},{:.1})",
                mouse_pos.0,
                mouse_pos.1
            ));
        });
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    // Setup logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Trace)
        .filter_module("gfx_backend_metal", log::LevelFilter::Error)
        .filter_module("rendy", log::LevelFilter::Error)
        .filter_module("rendy_imgui::gui", log::LevelFilter::Trace)
        //.filter_module("gui", log::LevelFilter::Trace)
        .init();

    let mut resources = Resources::new();

    // Use default rendy configuration.. this allows to inject device, heap, and queue
    // selection
    let config: Config = Default::default();

    // The factory is high-level owner of the device, instance, and manges memory, resources and
    // queue families.
    // Families represents the queue families on the device
    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    // Take care of winit launching windows
    let mut event_loop = EventsLoop::new();
    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();

    // Drain the event queue from winit
    event_loop.poll_events(|_| ());

    let mut imgui = init_imgui(&window);

    resources.insert(imgui);

    // Now that we have a window, we can create a surface
    let window: std::sync::Arc<winit::Window> = window.into();
    let surface = factory.create_surface(window.clone());

    let mut graph = init_render_graph(&mut factory, &mut families, surface, &resources);

    run(
        &mut event_loop,
        &mut factory,
        &mut families,
        &mut graph,
        window,
        &mut resources,
    )
    .unwrap();

    graph.dispose(&mut factory, &resources);
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
