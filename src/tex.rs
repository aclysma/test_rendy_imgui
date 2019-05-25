//!
//! A simple sprite example.
//! This examples shows how to render a sprite on a white background.
//!

#![cfg_attr(
not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
allow(unused)
)]

use {
    gfx_hal::Device as _,
    rendy::{
        command::{Families, QueueId, RenderPassEncoder},
        factory::{Config, Factory, ImageState},
        graph::{
            present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer,
            NodeImage,
        },
        memory::Dynamic,
        mesh::PosTex,
        resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
        shader::{ShaderKind, SourceLanguage, StaticShaderInfo},
        texture::Texture,
    },
};

#[cfg(feature = "spirv-reflection")]
use rendy::shader::SpirvReflection;

#[cfg(not(feature = "spirv-reflection"))]
use rendy::mesh::AsVertex;

use winit::{EventsLoop, WindowBuilder};

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/sprite.vert"),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets/sprite.frag"),
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

#[derive(Debug, Default)]
pub struct SpriteGraphicsPipelineDesc;

impl<B, T> SimpleGraphicsPipelineDesc<B, T> for SpriteGraphicsPipelineDesc
where
    B: gfx_hal::Backend,
    T: ?Sized,
{
    type Pipeline = SpriteGraphicsPipeline<B>;

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &T) -> rendy::shader::ShaderSet<B> {
        SHADERS.build(factory).unwrap()
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::InstanceRate,
    )> {
        #[cfg(feature = "spirv-reflection")]
        return vec![SHADER_REFLECTION
            .attributes_range(..)
            .unwrap()
            .gfx_vertex_input_desc(0)];

        #[cfg(not(feature = "spirv-reflection"))]
        return vec![PosTex::vertex().gfx_vertex_input_desc(0)];
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
                ],
            }],
            push_constants: Vec::new(),
        };
    }

    fn build<'b>(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<SpriteGraphicsPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        // This is how we can load an image and create a new texture.
        let image_reader = std::io::BufReader::new(std::fs::File::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/logo.png"
        ))?);

        let texture_builder =
            rendy::texture::image::load_from_image(image_reader, Default::default())?;

        let texture = texture_builder
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

        let descriptor_set = factory
            .create_descriptor_set(set_layouts[0].clone())
            .unwrap();

        unsafe {
            factory.device().write_descriptor_sets(vec![
                gfx_hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![gfx_hal::pso::Descriptor::Image(
                        texture.view().raw(),
                        gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                    )],
                },
                gfx_hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 1,
                    array_offset: 0,
                    descriptors: vec![gfx_hal::pso::Descriptor::Sampler(texture.sampler().raw())],
                },
            ]);
        }

        #[cfg(feature = "spirv-reflection")]
        let vbuf_size = SHADER_REFLECTION.attributes_range(..).unwrap().stride as u64 * 6;

        #[cfg(not(feature = "spirv-reflection"))]
        let vbuf_size = PosTex::vertex().stride as u64 * 6;

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
            // Fresh buffer.
            factory
                .upload_visible_buffer(
                    &mut vbuf,
                    0,
                    &[
                        PosTex {
                            position: [-0.5, 0.33, 0.0].into(),
                            tex_coord: [0.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, 0.33, 0.0].into(),
                            tex_coord: [1.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, -0.33, 0.0].into(),
                            tex_coord: [1.0, 0.0].into(),
                        },
                        PosTex {
                            position: [-0.5, 0.33, 0.0].into(),
                            tex_coord: [0.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, -0.33, 0.0].into(),
                            tex_coord: [1.0, 0.0].into(),
                        },
                        PosTex {
                            position: [-0.5, -0.33, 0.0].into(),
                            tex_coord: [0.0, 0.0].into(),
                        },
                    ],
                )
                .unwrap();
        }

        Ok(SpriteGraphicsPipeline {
            texture,
            vbuf,
            descriptor_set,
        })
    }
}

#[derive(Debug)]
pub struct SpriteGraphicsPipeline<B: gfx_hal::Backend> {
    texture: Texture<B>,
    vbuf: Escape<Buffer<B>>,
    descriptor_set: Escape<DescriptorSet<B>>,
}

impl<B, T> SimpleGraphicsPipeline<B, T> for SpriteGraphicsPipeline<B>
where
    B: gfx_hal::Backend,
    T: ?Sized,
{
    type Desc = SpriteGraphicsPipelineDesc;

    fn prepare(
        &mut self,
        _factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        _aux: &T,
    ) -> PrepareResult {
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            std::iter::once(self.descriptor_set.raw()),
            std::iter::empty::<u32>(),
        );
        encoder.bind_vertex_buffers(0, Some((self.vbuf.raw(), 0)));
        encoder.draw(0..6, 0..1);
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}
