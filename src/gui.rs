//!
//! The mighty imgui example.
//!

#![cfg_attr(
not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
allow(unused)
)]

use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory, ImageState},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    memory::Dynamic,
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    shader::{ShaderKind, SourceLanguage, StaticShaderInfo},
    texture::{image::TextureKind, Texture},
    util::types::vertex,
};

use shred::Resources;

use imgui::ImDrawVert;

use nalgebra_glm as glm;

#[cfg(feature = "spirv-reflection")]
use rendy::shader::SpirvReflection;

#[cfg(not(feature = "spirv-reflection"))]
use vertex::AsVertex;

lazy_static::lazy_static! {
    static ref VERTEX: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/gui.vert"),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/gui.frag"),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

#[cfg(feature = "spirv-reflection")]
lazy_static::lazy_static! {
    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

/// Type for position attribute of vertex.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Position2(pub [f32; 2]);
impl<T> From<T> for Position2
where
    T: Into<[f32; 2]>,
{
    fn from(from: T) -> Self {
        Position2(from.into())
    }
}
impl vertex::AsAttribute for Position2 {
    const NAME: &'static str = "position2";
    const FORMAT: gfx_hal::format::Format = gfx_hal::format::Format::Rg32Sfloat;
}

/// Type for color attribute of vertex.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PackedColorU32(pub [u8; 4]);
impl<T> From<T> for PackedColorU32
where
    T: Into<[u8; 4]>,
{
    fn from(from: T) -> Self {
        PackedColorU32(from.into())
    }
}
impl vertex::AsAttribute for PackedColorU32 {
    const NAME: &'static str = "packed_color_u32";
    const FORMAT: gfx_hal::format::Format = gfx_hal::format::Format::Rgba8Unorm;
}

// Format of an imgui vertex
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PosTexColor {
    /// Position of the vertex in 3D space.
    pub position: Position2,
    /// UV texture coordinates used by the vertex.
    pub tex_coord: vertex::TexCoord,
    /// RGBA color value of the vertex.
    pub color: PackedColorU32,
}

#[cfg(not(feature = "spirv-reflection"))]
impl AsVertex for PosTexColor {
    fn vertex() -> vertex::VertexFormat {
        vertex::VertexFormat::new((
            Position2::vertex(),
            vertex::TexCoord::vertex(),
            PackedColorU32::vertex(),
        ))
    }
}

// This defines a view-projection matrix
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct UniformArgs {
    pub mvp: nalgebra::Matrix4<f32>,
}

impl UniformArgs {
    fn new(frame_size: &imgui::FrameSize) -> UniformArgs {
        // DETERMINE VIEW MATRIX (just once)
        let view = glm::look_at_rh(
            &glm::make_vec3(&[0.0, 0.0, 5.0]),
            &glm::make_vec3(&[0.0, 0.0, 0.0]),
            &glm::make_vec3(&[0.0, 1.0, 0.0]).normalize(),
        );

        let projection = glm::ortho_rh_zo(
            0.0,
            frame_size.logical_size.0 as f32,
            0.0,
            frame_size.logical_size.1 as f32,
            -100.0,
            100.0,
        );

        // COMBINE THE VIEW AND PROJECTION MATRIX AHEAD OF TIME (just once)
        let mvp = projection * view;

        return UniformArgs { mvp };
    }
}

const UNIFORM_SIZE: u64 = std::mem::size_of::<UniformArgs>() as u64;

// For publishing the frame count as a resource
pub struct SwapchainBackbufferCount {
    backbuffer_count: u32,
}

impl SwapchainBackbufferCount {
    pub fn new(backbuffer_count: u32) -> Self {
        return SwapchainBackbufferCount { backbuffer_count };
    }

    pub fn get_backbuffer_count(&self) -> u32 {
        return self.backbuffer_count;
    }
}

#[derive(Debug, Default)]
pub struct ImguiRenderPipelineDesc;

impl<B> SimpleGraphicsPipelineDesc<B, Resources> for ImguiRenderPipelineDesc
where
    B: gfx_hal::Backend,
{
    type Pipeline = ImguiRenderPipeline<B>;

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        //TODO: imgui should be stenciled
        None
    }

    fn load_shader_set(
        &self,
        factory: &mut Factory<B>,
        _aux: &Resources,
    ) -> rendy::shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::VertexInputRate,
    )> {
        #[cfg(feature = "spirv-reflection")]
        return vec![SHADER_REFLECTION
            .attributes_range(..)
            .unwrap()
            .gfx_vertex_input_desc(0)];

        #[cfg(not(feature = "spirv-reflection"))]
        return vec![PosTexColor::vertex().gfx_vertex_input_desc(gfx_hal::pso::VertexInputRate::Vertex)];
    }

    fn layout(&self) -> Layout {
        #[cfg(feature = "spirv-reflection")]
        return SHADER_REFLECTION.layout().unwrap();

        #[cfg(not(feature = "spirv-reflection"))]
        return Layout {
            sets: vec![SetLayout {
                bindings: vec![
                    gfx_hal::pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: gfx_hal::pso::DescriptorType::SampledImage,
                        count: 1,
                        stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    gfx_hal::pso::DescriptorSetLayoutBinding {
                        binding: 1,
                        ty: gfx_hal::pso::DescriptorType::Sampler,
                        count: 1,
                        stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: false,
                    },
                    gfx_hal::pso::DescriptorSetLayoutBinding {
                        binding: 2, // does this binding ID matter? I think it does because binding 0 here steps on the sample image above
                        ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                        count: 1,
                        stage_flags: gfx_hal::pso::ShaderStageFlags::VERTEX,
                        immutable_samplers: false,
                    },
                ],
            }],
            push_constants: vec![],
        };
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &Resources,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<ImguiRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        log::trace!("DESC BUILD");

        let backbuffer_count = aux.fetch::<SwapchainBackbufferCount>().backbuffer_count as usize;
        assert_ne!(backbuffer_count, 0);

        let mut imgui = aux.fetch_mut::<imgui::ImGui>();
        let texture = imgui.prepare_texture(|texture_handle| {
            let kind = gfx_hal::image::Kind::D2(texture_handle.width, texture_handle.height, 1, 1);

            let view_kind = gfx_hal::image::ViewKind::D2;

            let sampler_info = gfx_hal::image::SamplerInfo::new(
                gfx_hal::image::Filter::Linear,
                gfx_hal::image::WrapMode::Clamp,
            );

            let texture_builder = rendy::texture::TextureBuilder::new()
                .with_raw_data(texture_handle.pixels, gfx_hal::format::Format::Rgba8Srgb)
                //no swizzle
                .with_data_width(texture_handle.width)
                .with_data_height(texture_handle.height)
                .with_kind(kind)
                .with_view_kind(view_kind)
                .with_sampler_info(sampler_info);

            return texture_builder
                .build(
                    ImageState {
                        queue,
                        stage: gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
                        access: gfx_hal::image::Access::SHADER_READ,
                        layout: gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                    },
                    factory,
                )
                .unwrap();
        });

        let mut uniform_buffers = vec![];
        let mut descriptor_sets = vec![];
        let mut draw_list_vbufs = vec![];
        let mut draw_list_ibufs = vec![];

        // Create uniform buffers and descriptor sets for each backbuffer, since this data can
        // change frame-to-frame
        for frame_index in 0..backbuffer_count {
            let mut uniform_buffer = factory
                .create_buffer(
                    BufferInfo {
                        size: UNIFORM_SIZE,
                        usage: gfx_hal::buffer::Usage::UNIFORM,
                    },
                    Dynamic,
                )
                .unwrap();

            let descriptor_set = factory
                .create_descriptor_set(set_layouts[0].clone())
                .unwrap();

            uniform_buffers.push(uniform_buffer);
            descriptor_sets.push(descriptor_set);
            draw_list_vbufs.push(vec![]);
            draw_list_ibufs.push(vec![]);
        }

        // Write descriptor sets for each backbuffer
        unsafe {
            for frame_index in 0..backbuffer_count {
                use gfx_hal::device::Device;
                factory.device().write_descriptor_sets(vec![
                    gfx_hal::pso::DescriptorSetWrite {
                        set: descriptor_sets[frame_index].raw(),
                        binding: 0,
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Image(
                            texture.view().raw(),
                            gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                        )],
                    },
                    gfx_hal::pso::DescriptorSetWrite {
                        set: descriptor_sets[frame_index].raw(),
                        binding: 1,
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Sampler(
                            texture.sampler().raw(),
                        )],
                    },
                    gfx_hal::pso::DescriptorSetWrite {
                        set: descriptor_sets[frame_index].raw(),
                        binding: 0, // Does this binding ID matter?
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Buffer(
                            uniform_buffers[frame_index].raw(),
                            Some(0)..Some(UNIFORM_SIZE),
                        )],
                    },
                ]);
            }
        }

        let quad_vertex_data = [
            PosTexColor {
                position: [-0.5, 0.33].into(),
                tex_coord: [0.0, 1.0].into(),
                color: [255, 255, 255, 255].into(),
            },
            PosTexColor {
                position: [0.5, 0.33].into(),
                tex_coord: [1.0, 1.0].into(),
                color: [255, 255, 255, 255].into(),
            },
            PosTexColor {
                position: [0.5, -0.33].into(),
                tex_coord: [1.0, 0.0].into(),
                color: [255, 255, 255, 255].into(),
            },
            PosTexColor {
                position: [-0.5, 0.33].into(),
                tex_coord: [0.0, 1.0].into(),
                color: [255, 255, 255, 255].into(),
            },
            PosTexColor {
                position: [0.5, -0.33].into(),
                tex_coord: [1.0, 0.0].into(),
                color: [255, 255, 255, 255].into(),
            },
            PosTexColor {
                position: [-0.5, -0.33].into(),
                tex_coord: [0.0, 0.0].into(),
                color: [255, 255, 255, 255].into(),
            },
        ];

        #[cfg(feature = "spirv-reflection")]
        let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride as u64
            * quad_vertex_data.len() as u64;

        #[cfg(not(feature = "spirv-reflection"))]
        let vbuf_size = PosTexColor::vertex().stride as u64 * quad_vertex_data.len() as u64;

        let mut quad_vbuf = factory
            .create_buffer(
                BufferInfo {
                    size: vbuf_size,
                    usage: gfx_hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        unsafe {
            // Fresh buffer.
            factory
                .upload_visible_buffer(&mut quad_vbuf, 0, &quad_vertex_data)
                .unwrap();
        }

        Ok(ImguiRenderPipeline {
            texture,
            quad_vbuf,
            draw_list_vbufs,
            draw_list_ibufs,
            descriptor_sets,
            uniform_buffers,
        })
    }
}

#[derive(Debug)]
pub struct ImguiRenderPipeline<B: gfx_hal::Backend> {
    texture: Texture<B>,
    quad_vbuf: Escape<Buffer<B>>,
    uniform_buffers: Vec<Escape<Buffer<B>>>,
    descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    draw_list_vbufs: Vec<Vec<Escape<Buffer<B>>>>,
    draw_list_ibufs: Vec<Vec<Escape<Buffer<B>>>>,
}

impl<B> SimpleGraphicsPipeline<B, Resources> for ImguiRenderPipeline<B>
where
    B: gfx_hal::Backend,
{
    type Desc = ImguiRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &Resources,
    ) -> PrepareResult {
        log::trace!("prepare");

        //TODO: Reuse these instead of dropping them every frame
        let draw_list_vbufs = &mut self.draw_list_vbufs[index];
        let draw_list_ibufs = &mut self.draw_list_ibufs[index];
        let uniform_buffers = &mut self.uniform_buffers[index];

        draw_list_vbufs.clear();
        draw_list_ibufs.clear();

        let ui = unsafe {
            if let Some(ui) = imgui::Ui::current_ui() {
                let ui = ui as *const imgui::Ui;
                let ui = ui.read();

                // Used for the view/projection
                let frame_size = ui.frame_size();
                let uniform_args = UniformArgs::new(&frame_size);

                factory
                    .upload_visible_buffer(uniform_buffers, 0, &[uniform_args])
                    .unwrap();

                ui.render::<_, ()>(|ui, draw_lists| {
                    for draw_list in &draw_lists {
                        // VERTEX BUFFER
                        let vertex_count = draw_list.vtx_buffer.len() as u64;

                        #[cfg(feature = "spirv-reflection")]
                        let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride
                            as u64
                            * vertex_count;

                        #[cfg(not(feature = "spirv-reflection"))]
                        let vbuf_size = PosTexColor::vertex().stride as u64 * vertex_count;

                        let mut vbuf = factory
                            .create_buffer(
                                BufferInfo {
                                    size: vbuf_size,
                                    usage: gfx_hal::buffer::Usage::VERTEX,
                                },
                                Dynamic,
                            )
                            .unwrap();

                        unsafe {
                            factory
                                .upload_visible_buffer(&mut vbuf, 0, &draw_list.vtx_buffer)
                                .unwrap();
                        }

                        draw_list_vbufs.push(vbuf);

                        //INDEX BUFFER
                        let ibuf_size =
                            draw_list.idx_buffer.len() as u64 * std::mem::size_of::<u16>() as u64;
                        let mut ibuf = factory
                            .create_buffer(
                                BufferInfo {
                                    size: ibuf_size,
                                    usage: gfx_hal::buffer::Usage::INDEX,
                                },
                                Dynamic,
                            )
                            .unwrap();

                        unsafe {
                            factory
                                .upload_visible_buffer(&mut ibuf, 0, &draw_list.idx_buffer)
                                .unwrap();
                        }

                        draw_list_ibufs.push(ibuf);
                    }

                    return Ok(());
                });
            }
        };

        return PrepareResult::DrawRecord;
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _aux: &Resources,
    ) {
        log::trace!("draw");

        let draw_list_vbufs = &self.draw_list_vbufs[index];
        let draw_list_ibufs = &self.draw_list_ibufs[index];

        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            std::iter::once(self.descriptor_sets[index].raw()),
            std::iter::empty::<u32>(),
        );

        encoder.bind_vertex_buffers(0, Some((self.quad_vbuf.raw(), 0)));
        encoder.draw(0..6, 0..1);

        unsafe {
            use imgui::sys;
            use imgui::sys::ImDrawData;

            let draw_data = imgui::sys::igGetDrawData();
            if !draw_data.is_null() {
                let draw_lists = (*draw_data).cmd_lists().iter().map(|&ptr| unsafe {
                    imgui::DrawList {
                        cmd_buffer: (*ptr).cmd_buffer.as_slice(),
                        idx_buffer: (*ptr).idx_buffer.as_slice(),
                        vtx_buffer: (*ptr).vtx_buffer.as_slice(),
                    }
                });

                //TODO: Verify the draw list index doesn't exceed the vbuf/ibuf list
                let mut draw_list_index = 0;
                for draw_list in draw_lists {
                    encoder
                        .bind_vertex_buffers(0, Some((draw_list_vbufs[draw_list_index].raw(), 0)));

                    encoder.bind_index_buffer(
                        draw_list_ibufs[draw_list_index].raw(),
                        0,
                        gfx_hal::IndexType::U16,
                    );

                    let mut element_begin_index = 0;
                    for cmd in draw_list.cmd_buffer {
                        let element_end_index = element_begin_index + cmd.elem_count;
                        encoder.draw_indexed(element_begin_index..element_end_index, 0, 0..1);

                        element_begin_index = element_end_index;
                    }

                    draw_list_index += 1;
                }
            }
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Resources) {}
}
