# Hailo PCIe DKMS patch — `mmap_read_lock` around `find_vma`

## What this fixes

Stock HailoRT 4.23.0 PCIe driver (`hailort-pcie-driver_4.23.0_all.deb`, downloaded from the Hailo Developer Zone) calls `find_vma` inside `hailo_vdma_buffer_map` **without holding `mmap_read_lock`**. On Linux kernel **6.12 or newer**, `find_vma` enforces that the caller holds the mmap lock, and an unlocked call results in:

- A 30+ minute hang on the first inference call after a fresh boot.
- No recovery short of a hard reboot — the device char node `/dev/hailo0` is wedged, the VDMA path won't release.
- Reproducible on every kernel ≥ 6.12 (we observed it on Ubuntu 24.04.4 HWE 6.17.0-20-generic).

The fix is a 2-line wrap around the `find_vma` call. The same change has already landed on Hailo's `master` branch via [PR #26](https://github.com/hailo-ai/hailort-drivers/pull/26) but as of 2026-04-28 has **not been backported to the `hailo8` branch** that the apps-infrastructure deb ships from. PR #44 is open upstream tracking this backport.

Until Hailo merges the backport, every Hailo-8 / Hailo-8L user on a kernel-6.12+ host needs the patch applied locally. This directory packages it for DKMS so the patch survives every kernel upgrade automatically.

## How to apply

Tested on Ubuntu 24.04.4 LTS, kernel 6.17 HWE, HailoRT 4.23.0.

```bash
# 1. Place the patch under the DKMS source tree
sudo mkdir -p /usr/src/hailo_pci-4.23.0/patches/
sudo cp 0001-mmap-read-lock-around-find-vma.patch \
        /usr/src/hailo_pci-4.23.0/patches/

# 2. Register the patch in dkms.conf — MUST be on its own line
echo 'PATCH[0]="0001-mmap-read-lock-around-find-vma.patch"' \
  | sudo tee -a /usr/src/hailo_pci-4.23.0/dkms.conf

# 3. Rebuild + reinstall the module against the running kernel
sudo dpkg-reconfigure hailort-pcie-driver

# 4. Verify
sudo dkms status hailo_pci
ls /dev/hailo*                          # expect /dev/hailo0
hailortcli fw-control identify          # expect Board=Hailo-8, Firmware=4.23.0
```

### Critical detail

The `PATCH[0]=…` line **must be on its own line** in `dkms.conf`. An earlier failure mode of ours: appending it via `tee -a` without a trailing newline merged it onto `AUTOINSTALL=yes` and DKMS silently ignored the patch — you get the same hang as if the patch wasn't applied. Always re-read the file after editing:

```bash
grep -n PATCH /usr/src/hailo_pci-4.23.0/dkms.conf
```

You should see `PATCH[0]=...` on a line of its own.

## Lifecycle

- **On kernel upgrade:** DKMS runs `dkms install` automatically and re-applies the patch. No manual intervention required.
- **On HailoRT upgrade:** if the new deb's source tree drops or replaces the patches/ subdir, re-run the apply steps above.
- **When Hailo ships the fix in the deb (4.23.1 or `hailo8`-branch update):** remove the patch + re-run dpkg-reconfigure:

  ```bash
  sudo rm /usr/src/hailo_pci-4.23.0/patches/0001-mmap-read-lock-around-find-vma.patch
  sudo sed -i '/^PATCH\[0\]=/d' /usr/src/hailo_pci-4.23.0/dkms.conf
  sudo dpkg-reconfigure hailort-pcie-driver
  ```

## The patch itself

```diff
diff --git a/linux/vdma/memory.c b/linux/vdma/memory.c
@@ -167,7 +167,9 @@ struct hailo_vdma_buffer *hailo_vdma_buffer_map(struct device *dev,
     }

     if (HAILO_DMA_DMABUF_BUFFER != buffer_type) {
+        mmap_read_lock(current->mm);
         vma = find_vma(current->mm, addr_or_fd);
+        mmap_read_unlock(current->mm);
         if (IS_ENABLED(HAILO_SUPPORT_MMIO_DMA_MAPPING)) {
             if (NULL == vma) {
                 dev_err(dev, "no vma for virt_addr/size = 0x%08lx/0x%08zx\n", addr_or_fd, size);
```

Two lines. Acquires the read side of the process's mmap lock for the duration of the `find_vma` lookup — exactly the locking contract the post-6.12 kernel API requires.

## Provenance

- Technique sourced from the master-branch fix at `hailo-ai/hailort-drivers` PR #26.
- Repackaged for DKMS by Daniel during the OpenClaw Hailo-8L bring-up (Dell OptiPlex 7060 SFF, Ubuntu 24.04, kernel 6.17 HWE, HailoRT 4.23.0).
- Verified end-to-end on 2026-04-23: 210-thumbnail TinyCLIP embedding workload completed in 22 s with the patch, hung indefinitely without it.

## License

Public-domain / CC0 for the patch text itself. The original `linux/vdma/memory.c` is licensed under Hailo's source license — see `hailort-drivers` upstream. This repackaging adds no new copyrightable expression.
